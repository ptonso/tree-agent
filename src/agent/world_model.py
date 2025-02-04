import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from src.agent.structures import State, Observation
from src.agent.autoencoder import Encoder, Decoder
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

        self.critic = critic

        self.state_dim = wm_cfg.latent_dim

        self.beta_pred = wm_cfg.beta_pred
        self.beta_kl = wm_cfg.beta_kl
        self.beta_reward = wm_cfg.beta_reward
        self.beta_value = wm_cfg.beta_value
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

        self.reward_head = nn.Linear(wm_cfg.latent_dim , 1)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.agent.world_model.cnn_lr,
            eps=1e-20
        )

    def encode(self, observation: Observation) -> tuple:
        """
        Returns:
        - State containing (B, E*2)
        """
        mu, logvar = self.encoder(observation.as_tensor).chunk(2, dim=1)
        # latent = torch.cat([mu, logvar], dim=1) # (B, E*2)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Full β-VAE pass.
        Args:
            observation: (B, C, H, W) raw input images.
        """
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        reward_hat = self.reward_head(z).squeeze(-1)
        return x_hat, mu, logvar, z, reward_hat
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # sample noise
        return mu + eps * std

    def loss_function(
            self, x: torch.Tensor, x_hat: torch.Tensor, 
            mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor,
            reward_hat: torch.Tensor, reward: torch.Tensor,
            returns: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute β-VAE loss:
        - Reconstruction loss (MSE).
        - KL divergence loss (with free nats).
        """
        symlog_x = self.symlog(x)
        symlog_x_hat = self.symlog(x_hat)

        symlog_reward = self.symlog(reward)
        symlog_reward_hat = self.symlog(reward_hat)

        pred_loss = F.mse_loss(symlog_x_hat, symlog_x, reduction='mean')
        reward_loss = F.mse_loss(symlog_reward_hat, symlog_reward, reduction='mean')

        kl_beta, kl_loss = self.compute_kl_loss(mu, logvar)

        triplet_loss = self.compute_triplet_loss(z, returns)

        mu_recons, logvar_recons = self.encoder(x_hat).chunk(2, dim=1)

        with torch.no_grad():
            v_original = self.critic(z).detach()
            z_recons = self.reparametrize(mu_recons, logvar_recons)
            v_reconstructed = self.critic(z_recons).detach()

        value_loss = F.mse_loss(v_reconstructed, v_original, reduction='mean')

        total_loss = (
            self.beta_pred * pred_loss
            + self.beta_reward * reward_loss
            + kl_beta * kl_loss
            + self.beta_value * value_loss
            + self.beta_triplet * triplet_loss
            )
        return total_loss, pred_loss, kl_loss, reward_loss, value_loss, triplet_loss
    
    def symlog(self, x: torch.Tensor) -> torch.Tensor:
        """apply symlog transformation. (compress large values)"""
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        free_nats = self.config.agent.world_model.free_nats
        kl_loss = torch.mean(torch.clamp(kl_per_sample, min=free_nats))
        self.kl_ema = 0.99 * self.kl_ema + 0.01 * kl_loss.item()
        kl_beta = self.beta_kl * torch.exp(self.kl_scale * (kl_loss - self.kl_target))
        kl_beta = torch.clamp(kl_beta, 0.001, 1.0)
        return kl_beta, kl_loss
    
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


    def train_step(
            self,
            trajectories: List['Trajectory'],
            logger: Optional[object] = None,
        ):
        """Train world model using list of trajectory"""

        mb_size = self.config.agent.world_model.mb_size
        n_epochs = self.config.agent.world_model.n_epochs

        batch = Batch(trajectories, self.device)
        obs_tensor = batch.tensors.observations # (B*T, C, H, W)
        reward_tensor = batch.tensors.rewards   # (B*T,)
        returns_tensor = batch.tensors.returns  # (B*T,)
        n_samples = obs_tensor.size(0)

        total_loss_sum, recon_loss_sum, kl_loss_sum, reward_loss_sum, value_loss_sum, triplet_loss_sum = 0., 0., 0., 0., 0., 0.
        total_steps = 0

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples, device=self.device)

            # Mini-batch training
            for start in range(0, obs_tensor.size(0), mb_size):
                end = min(start + mb_size, obs_tensor.size(0))
                batch_idx = indices[start:end]
                obs_batch = obs_tensor[batch_idx]
                reward_batch = reward_tensor[batch_idx]
                returns_batch = returns_tensor[batch_idx]

                x_hat, mu, logvar, z, reward_hat = self.forward(obs_batch)

                loss, recon_loss, kl_loss, reward_loss, value_loss, triplet_loss = self.loss_function(obs_batch, x_hat, mu, logvar, z, reward_hat, reward_batch, returns_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = (end - start)
                total_loss_sum += loss.item() * batch_size
                recon_loss_sum += recon_loss.item() * batch_size
                kl_loss_sum += kl_loss.item() * batch_size
                reward_loss_sum += reward_loss.item() * batch_size
                value_loss_sum += value_loss.item() * batch_size
                triplet_loss_sum += triplet_loss.item() * batch_size
                total_steps += batch_size

        avg_total_loss = total_loss_sum / total_steps
        avg_recon_loss = recon_loss_sum / total_steps
        avg_kl_loss = kl_loss_sum / total_steps
        avg_reward_loss = reward_loss_sum / total_steps
        avg_value_loss = value_loss_sum / total_steps
        avg_triplet_loss = triplet_loss_sum / total_steps

        if logger:
            logger.info(f"Recon Loss:   {avg_recon_loss:.4f}")
            logger.info(f"KL Loss:      {avg_kl_loss:.4f}")
            logger.info(f"Reward Loss:  {avg_reward_loss:.4f}")
            logger.info(f"Value Loss:   {avg_value_loss:.4f}")
            logger.info(f"Triplet Loss: {avg_triplet_loss:.4f}")


        return avg_total_loss, avg_recon_loss, avg_kl_loss, avg_reward_loss, avg_value_loss, avg_triplet_loss

