import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Optional, Tuple, List

import time

from src.agent.structures import State, Observation
from src.agent.vae import Encoder, Decoder
from src.agent.transition_model import TransitionModel
from src.agent.mlp import MLP
from src.agent.agent import Critic
from src.agent.batch import Batch

class WorldModel(nn.Module):
    def __init__(self, config: object, critic: Critic):
        """
        output_multiplier: 1 for standard autoencoder, 2 for VAE
        """
        super().__init__()
        self.config = config
        self.device = config.device
        wm_cfg = config.agent.world_model

        self.state_dim = wm_cfg.latent_dim

        self.beta_recon = wm_cfg.beta_recon
        self.beta_kl = wm_cfg.beta_kl

        self.beta_reward = wm_cfg.beta_reward
        self.beta_returns = wm_cfg.beta_returns
        self.beta_triplet = wm_cfg.beta_triplet

        self.kl_target = 3.0
        self.kl_scale = 0.1 # scale factor for adative KL
        self.kl_ema = 3.0   # moving average for kl

        self.encoder = Encoder(
            in_channels=config.env.channels,
            in_height=config.env.height,
            in_width=config.env.width,
            latent_dim=wm_cfg.latent_dim,
            cfg=wm_cfg.encoder
        )

        self.decoder = Decoder(
            out_channels=config.env.channels,
            out_height=config.env.height,
            out_width=config.env.width,
            latent_dim=wm_cfg.latent_dim,
            cfg=wm_cfg.decoder
        )

        self.reward_predictor = MLP(
            input_dim=wm_cfg.latent_dim,
            output_dim=1,
            hidden_layers=[128, 128]
        )

        self.vae_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) + 
            list(self.reward_predictor.parameters()),
            lr=wm_cfg.vae_lr,
            eps=1e-20
        )


        self.use_transition = False

        if self.use_transition:
            self.transition_model = TransitionModel(
                latent_dim=wm_cfg.latent_dim,
                action_dim=1,
                cfg=wm_cfg.transition,
                device=self.device
            )

            self.transition_optimizer = torch.optim.Adam(
                self.transition_model.parameters(),
                lr=wm_cfg.transition.lr,
                eps=1e-20
            )


    def encode(self, observation: Observation) -> tuple:
        """
        Returns:
        - State containing (B, E*2)
        """
        mu, logvar = self.encoder(observation.as_tensor).chunk(2, dim=1)
        z = self.reparametrize(mu, logvar) # (B, E)
        return State.from_encoder(
            z.cpu().detach().numpy(), mu, logvar, device=self.device
            )

    def decode(self, state: State) -> torch.Tensor:
        """
        Decodes latent vector into an image.
        """
        mu, logvar = state.mu_logvar
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return Observation.from_decoder(x_hat, device=self.device)
    

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def compute_vae_loss(
            self, x, reward,
            returns,
        ) -> Tuple[torch.Tensor, ...]:
        """
        Compute β-VAE loss:
        - Reconstruction loss (MSE).
        - KL divergence loss (with free nats).
        """
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)

        symlog_x, symlog_x_hat = self.symlog(x), self.symlog(x_hat)
        recon_loss = F.mse_loss(symlog_x_hat, symlog_x, reduction='mean')

        kl_beta, kl_loss = self.compute_kl_loss(mu, logvar)

        reward_loss = torch.zeros(1, device=self.device)
        returns_loss = torch.zeros(1, device=self.device)
        triplet_loss = torch.zeros(1, device=self.device)

        if self.beta_triplet > 0:
            triplet_loss = self.compute_triplet_loss(z, returns)

        if self.beta_reward > 0:
            reward_hat = self.reward_predictor(z)
            symlog_reward_hat = self.symlog(reward_hat).squeeze(-1)
            symlog_reward = self.symlog(reward)
            reward_loss = F.mse_loss(symlog_reward_hat, symlog_reward)

        if self.beta_returns > 0:
            returns_hat = self.reward_predictor(z)
            symlog_returns_hat = self.symlog(returns_hat).squeeze(-1)
            symlog_returns = self.symlog(returns)
            returns_loss = F.mse_loss(symlog_returns_hat, symlog_returns)
        
        vae_loss = (
            self.beta_recon * recon_loss
            + kl_beta * kl_loss
            + self.beta_reward * reward_loss

            + self.beta_returns * returns_loss
            + self.beta_triplet * triplet_loss
            )
        return vae_loss, recon_loss, kl_loss, reward_loss, returns_loss, triplet_loss, z
    

    def compute_transition_loss(self, z: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu_next, logvar_next = self.encoder(next_obs).chunk(2, dim=1)
            z_next_real = self.reparametrize(mu_next, logvar_next)
        z_next_pred = self.transition_model(z, action)

        transition_loss = F.mse_loss(z_next_pred, z_next_real)
        return transition_loss

    def symlog(self, x: torch.Tensor) -> torch.Tensor:
        """apply symlog transformation. (compress large values)"""
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        free_nats = self.config.agent.world_model.free_nats
        
        kl_loss = torch.mean(torch.clamp(kl_per_sample, min=free_nats))
               
        kl_beta = self.beta_kl * torch.exp(self.kl_scale * (kl_loss - self.kl_target))
        kl_beta = torch.clamp(kl_beta, 0.001, 1.0)
        return kl_beta, kl_loss

    def train_step(
            self,
            trajectories: List['Trajectory'],
            logger: Optional[object] = None,
        ):
        """Train world model using list of trajectory"""

        mb_size = self.config.agent.world_model.mb_size
        n_epochs = self.config.agent.world_model.n_epochs

        batch = Batch(trajectories, self.device)
        obs_tensor = batch.tensors.observations           # (B*T, C, H, W)
        next_obs_tensor = batch.tensors.next_observations # (B*T, C, H, W)
        reward_tensor = batch.tensors.rewards             # (B*T,)
        returns_tensor = batch.tensors.returns            # (B*T,)
        actions_tensor = batch.tensors.actions            # (B*T, *action_dim)
        n_samples = obs_tensor.size(0)

        kl_loss_accum = 0.0
        loss_sums = torch.zeros(7, device=self.device)
        total_steps = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples, device=self.device)

            # Mini-batch training
            for start in range(0, obs_tensor.size(0), mb_size):

                end = min(start + mb_size, obs_tensor.size(0))
                batch_size = (end - start)
                batch_idx = indices[start:end]
                obs_batch = obs_tensor[batch_idx]
                next_obs_batch = next_obs_tensor[batch_idx]
                reward_batch = reward_tensor[batch_idx]
                returns_batch = returns_tensor[batch_idx]
                action_batch = actions_tensor[batch_idx]

                vae_loss, recon_loss, kl_loss, rewards_loss, returns_loss, triplet_loss, z = \
                    self.compute_vae_loss(obs_batch, reward_batch, returns_batch)

                kl_loss_accum += kl_loss.detach().item() * batch_size


                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                self.vae_optimizer.step()

                transition_loss = torch.zeros(1, device=self.device)
                if self.use_transition:
                    transition_loss = \
                        self.compute_transition_loss(z.detach(), action_batch, next_obs_batch)
    
                    self.transition_optimizer.zero_grad()
                    transition_loss.backward()
                    self.transition_optimizer.step()

                with torch.no_grad():
                    loss_items = [vae_loss, recon_loss, kl_loss, rewards_loss, returns_loss, triplet_loss, transition_loss]
                    loss_tensors = [loss if loss.dim() == 0 else loss.squeeze() for loss in loss_items]
                    loss_values = torch.stack(loss_tensors)

                    loss_sums += loss_values * batch_size
                    total_steps += batch_size


        self.kl_ema = 0.99 * self.kl_ema + 0.01 * (kl_loss_accum / total_steps)

        (
            avg_vae_loss, avg_recon_loss, avg_kl_loss, 
            avg_rewards_loss, avg_returns_loss,
            avg_triplet_loss, avg_transition_loss 
        ) = (loss_sums / total_steps)


        if logger:
            logger.info(f"VAE Loss:        {avg_vae_loss:.4f}")
            logger.info(f"    Recon Loss:    {avg_recon_loss:.4f}")
            logger.info(f"    KL Loss:       {avg_kl_loss:.4f}")
            logger.info(f"    Rewards Loss:  {avg_rewards_loss:.4f}")
            logger.info(f"    Returns Loss:  {avg_returns_loss:.4f}")
            logger.info(f"    Triplet Loss:  {avg_triplet_loss:.4f}")
            logger.info(f"Transition Loss: {avg_transition_loss:.4f}")


        wm_metrics = {
            "avg_vae_loss": avg_vae_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_kl_loss": avg_kl_loss,
            "avg_rewards_loss": avg_rewards_loss,
            "avg_transition_loss": avg_transition_loss,
            "avg_returns_loss": avg_returns_loss,
            "avg_triplet_loss": avg_triplet_loss
            }
        wm_metrics = {key: value.detach().cpu().item() for key, value in wm_metrics.items()}
        return wm_metrics

# ---

    def compute_td_loss(self, z_t, z_next_real, actions, returns):
        z_next_pred = self.transition_model(torch.cat([z_t, actions], dim=-1))

        returns_diff = returns[:, 1:] - returns[:, :-1]
        td_gradient = torch.autograd.grad(returns_diff.sum(), z_t, create_graph=True)[0]

        td_loss = ((z_t - z_next_pred + 0.1 * td_gradient) ** 2).mean()

        return td_loss



    def compute_triplet_loss(self, embeds: torch.Tensor, returns: torch.Tensor, margin=0.2):
        batch_size = embeds.shape[0]

        min_return, max_return = returns.min(), returns.max()
        norm_returns = (returns - min_return) / (max_return - min_return + 1e-8)

        pos_indices = torch.zeros(batch_size, dtype=torch.long)
        neg_indices = torch.zeros(batch_size, dtype=torch.long)

        for i in range(batch_size):
            pos_mask = torch.abs(norm_returns - norm_returns[i]) < 0.1 # margin m+
            neg_mask = torch.abs(norm_returns - norm_returns[i]) > 0.3 # margin m-

            pos_candidates = torch.where(pos_mask)[0]
            neg_candidates = torch.where(neg_mask)[0]

            if len(pos_candidates) > 1:
                pos_indices[i] = pos_candidates[torch.randint(0, len(pos_candidates), (1,))]
            else:
                pos_indices[i] = i
            
            if len(neg_candidates) > 1:
                neg_indices[i] = neg_candidates[torch.randint(0, len(neg_candidates), (1,))]
            else:
                neg_indices[i] = i
            
        pos_embeds = embeds[pos_indices]
        neg_embeds = embeds[neg_indices]
        anchor_embeds = embeds

        pos_dist = F.pairwise_distance(anchor_embeds, pos_embeds, p=2)
        neg_dist = F.pairwise_distance(anchor_embeds, neg_embeds, p=2)
        
        triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0).mean()

        return triplet_loss