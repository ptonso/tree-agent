import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, List, Dict

from src.agent.structures import State, Observation
from src.agent.vae import Encoder, Decoder
from src.agent.transition_model import TransitionModel
from src.agent.mlp import MLP
from src.agent.agent import Actor
from src.agent.batch import Batch, BatchTensors

class WorldModel(nn.Module):
    def __init__(self, config: object, actor: Actor):
        super().__init__()
        self.config = config
        self.device = config.device
        self._init_hyperparameter(config.agent.world_model)
        self._init_networks(config)
        self.actor = actor

    def _init_hyperparameter(self, wm_cfg):
        self.state_dim = wm_cfg.latent_dim
        self.beta_pred = wm_cfg.beta_pred
        self.beta_dym = wm_cfg.beta_dym
        self.beta_rep = wm_cfg.beta_rep
        self.gamma = self.config.agent.gamma
        self.free_nats = wm_cfg.free_nats
        self.kl_target = 3.0
        self.kl_scale = 0.1 # scale factor for adative KL
        self.kl_ema = 3.0   # moving average for kl

    def _init_networks(self, config):
        wm_cfg = config.agent.world_model

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

        self.continue_predictor = MLP(
            input_dim=wm_cfg.latent_dim,
            output_dim=1,
            hidden_layers=[128, 128]
        )

        self.transition_model = TransitionModel(
            latent_dim=wm_cfg.latent_dim,
            action_dim=1,
            cfg=wm_cfg.transition,
            device=self.device
        )

        self.wm_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) + 
            list(self.reward_predictor.parameters()) +
            list(self.continue_predictor.parameters()) +
            list(self.transition_model.parameters()),
            lr=wm_cfg.lr,
            eps=1e-20
        )


    def encode(self, observation: Observation) -> tuple:
        """Encode observation into latent state representation"""
        mu, logvar = self.encoder(observation.as_tensor)
        z = self.reparametrize(mu, logvar)
        z_data = z.cpu().detach().numpy()
        return State.from_encoder(z_data, mu, logvar, device=self.device)

    def decode(self, state: State) -> torch.Tensor:
        """Decodes latent vector into an image."""
        x_hat = self.decoder(state.as_tensor)
        return Observation.from_decoder(x_hat, device=self.device)
    
    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict_future(self, z: torch.Tensor, action_seq: torch.Tensor, horizon: int):
        """Multi-step rollout based on action.
        Return tensors at [B, H, *dim]"""
        preds = defaultdict(list)
        current_z = z

        for t in range(horizon):
            action_t = action_seq[:, t]
            mu_hat, logvar_hat = self.transition_model(current_z, action_t)
            next_z = self.reparametrize(mu_hat, logvar_hat)

            reward_hat = self.reward_predictor(next_z)
            cont_hat = self.continue_predictor(next_z)

            preds["reward"].append(reward_hat)
            preds["continue"].append(cont_hat)
            preds["mu"].append(mu_hat)
            preds["logvar"].append(logvar_hat)
            preds["z_next"].append(next_z)
            current_z = next_z

        for key, value in preds.items():
            preds[key] = torch.stack(value, dim=1) # [B, H, *dim]
        return preds
        


    def compute_losses(self, mb_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute world_model loss:
        - Prediction loss
            - decoder:  symlog²
            - reward:   symlog²
            - continue: logistic
        - Dynamic loss
            - z_pred and z_real: KL div
        - Representation loss
            - ???
        """
        obs0 = mb_data['obs0']
        action_seq = mb_data['action_seq']
        reward_seq = mb_data['reward_seq'].unsqueeze(-1)     # [mb, horizon, 1]
        continue_seq = mb_data['continue_seq'].unsqueeze(-1) # [mb, horizon, 1]
        obs_seq = mb_data['obs_seq']                         # [mb*horizon, C, H, W]

        mb = obs0.shape[0]
        horizon = self.config.agent.world_model.horizon

        mu0, logvar0 = self.encoder(obs0) # (B*T, latent_dim)
        z0 = self.reparametrize(mu0, logvar0)

        preds = self.predict_future(z0, action_seq, horizon)

        symlog_reward_seq = self.symlog(reward_seq)
        reward_loss = F.mse_loss(preds['reward'], symlog_reward_seq, reduction='mean')
        continue_loss = F.binary_cross_entropy_with_logits(preds['continue'], continue_seq, reduction='mean')

        z_preds_flat = preds['z_next'].view(mb * horizon, -1)
        recon_obs = self.decoder(z_preds_flat)
        symlog_obs_seq = self.symlog(obs_seq)
        recon_loss = F.mse_loss(recon_obs, symlog_obs_seq, reduction='mean')

        mu_preds_flat = preds['mu'].view(mb * horizon, -1)
        logvar_preds_flat = preds['logvar'].view(mb * horizon, -1)
        with torch.no_grad():
            mu_next, logvar_next = self.encoder(obs_seq)

        # KL[ stopgrad(q) || p ]
        dyn_loss = self.gaussian_kl(
            mu_next.detach(), 
            logvar_next.detach(), 
            mu_preds_flat, 
            logvar_preds_flat, 
        )

        # KL[ q || stopgrad(p) ]
        rep_loss = self.gaussian_kl(
            mu_preds_flat.detach(), 
            logvar_preds_flat.detach(),
            mu_next, 
            logvar_next
        )

        kl_adj = self.kl_scale * (dyn_loss.detach() - self.kl_ema)
        self.kl_ema = 0.99 * self.kl_ema + 0.01 * dyn_loss.detach()
        
        total_loss = (
            self.beta_pred * (
                recon_loss 
                + reward_loss 
                + continue_loss
                )
            + (self.beta_dym + kl_adj) * dyn_loss
            + self.beta_rep * rep_loss
            )
        
        losses = {
            'total_loss': total_loss,
            'recon_loss': self.beta_pred * recon_loss,
            'reward_loss': self.beta_pred * reward_loss,
            'continue_loss': self.beta_pred * continue_loss,
            'dyn_loss': self.beta_dym * dyn_loss,
            'rep_loss': self.beta_rep * rep_loss
        }

        return losses

    def gaussian_kl(
            self, 
            mu_q: torch.Tensor, 
            logvar_q: torch.Tensor,
            mu_p: torch.Tensor,
            logvar_p: torch.Tensor,
            ):
        """KL betwene diagonal Gaussian q ~ N(mu_q, exp(logvar_q))
        and p ~ N(mu_p, exp(logvar_p))"""
        sigma_q2 = logvar_q.exp()
        sigma_p2 = logvar_p.exp()
        kl = 0.5 * (
            (sigma_q2 + (mu_q - mu_p).pow(2)) / sigma_p2
            + (logvar_p - logvar_q)
            - 1.
        )
        kl_per_sample = torch.clamp(kl, min=self.free_nats).sum(dim=-1)
        return kl_per_sample.mean() # sum over latent dim, average over batch


    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """Apply symlog transformation to compress large values"""
        return torch.sign(x) * torch.log1p(torch.abs(x))


    def train_step(
            self,
            trajectories: List['Trajectory'],
            logger: Optional[object] = None,
        ):
        """Train world model using list of trajectory"""
        mb_size = self.config.agent.world_model.mb_size
        n_epochs = self.config.agent.world_model.n_epochs
        horizon = self.config.agent.world_model.horizon
        
        batch = Batch(trajectories, self.device)
        tensors = batch.tensors

        metrics_accumulator = MetricAccumulator(self.device)

        for epoch in range(n_epochs):
                mb_size = self.config.agent.world_model.mb_size
                valid_indices = self._get_valid_indices(tensors, horizon)
                shufled_indices = valid_indices[torch.randperm(len(valid_indices))]

                for start_idx in range(0, len(shufled_indices), mb_size):
                    end_idx = min(start_idx + mb_size, len(shufled_indices))
                    mb_size = end_idx - start_idx
                    mb_indices = shufled_indices[start_idx:end_idx]
                    mb_data = self._get_minibatch_data(tensors, mb_indices, mb_size, horizon)

                    losses = self.compute_losses(mb_data)
                    self.wm_optimizer.zero_grad()
                    losses['total_loss'].backward()
                    self.wm_optimizer.step()

                    metrics_accumulator.update(losses, mb_size)

        final_metrics = metrics_accumulator.get_average()
        self._log_metrics(final_metrics, logger)
    
        return final_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], logger: Optional[object]):
        """Log training metrics if loger is provided"""
        if logger:
            logger.info(f"World Model Loss:    {metrics['total_loss']:.4f}")
            logger.info(f"    Recon Loss:      {metrics['recon_loss']:.4f}")
            logger.info(f"    Reward Loss:     {metrics['reward_loss']:.4f}")
            logger.info(f"    Continue Loss:   {metrics['continue_loss']:.4f}")
            logger.info(f"    Dyn Loss:        {metrics['dyn_loss']:.4f}")
            logger.info(f"    Rep Loss:        {metrics['rep_loss']:.4f}")


    def _get_valid_indices(self, tensors: BatchTensors, horizon: int) -> torch.Tensor:
        """
        Get indices of valid starting points for training sequences.
        A valid starting point must have at least 'horizon' steps remaining in the trajectory
        """
        traj_lengths = torch.tensor(tensors.trajectory_lengths, device=self.device)
        traj_ends = torch.cumsum(traj_lengths, dim=0)
        traj_starts = torch.cat((torch.tensor([0], device=self.device), traj_ends[:-1]))

        valid_starts = []
        for start, end in zip(traj_starts, traj_ends):
            if (end - start) > horizon:
                valid_starts.extend(range(start.item(), end.item() - horizon))
    
        return torch.tensor(valid_starts, device=self.device)


    def _get_minibatch_data(
            self,
            tensors: BatchTensors,
            indices: torch.Tensor,
            mb_size: int,
            horizon: int
        ) -> Dict[str, torch.Tensor]:
        """
        Extract minibatch data starting from given indices.
        """
        # Create indices for all timestemps in the sequence
        sequence_indices = indices.unsqueeze(1) + torch.arange(horizon, device=self.device).unsqueeze(0)
        flat_seq = sequence_indices.flatten() # => [B*horizon]

        obs0 = tensors.observations[indices]              # [mb_size, C, H, W]
        
        action_seq = tensors.actions[flat_seq].view(mb_size, horizon)
        reward_seq = tensors.rewards[flat_seq].view(mb_size, horizon)
        continue_seq = tensors.dones[flat_seq].view(mb_size, horizon)
        obs_seq = tensors.observations[flat_seq] # [mb_size*horizon, C, H, W]

        return {
            'obs0': obs0,
            'action_seq': action_seq,
            'reward_seq': reward_seq,
            'continue_seq': continue_seq,
            'obs_seq': obs_seq,
        }


class MetricAccumulator:
    """Helper class to accumulate and average training metrics"""
    def __init__(self, device: str):
            self.sums = {}
            self.counts = {}
            self.device = device

    def update(self, losses: Dict[str, torch.Tensor], batch_size: int):
        """Update metrics maintaining tensors on device"""
        with torch.no_grad():
            for name, value in losses.items():
                if name not in self.sums:
                    self.sums[name] = torch.zeros(1, device=self.device)
                    self.counts[name] = 0
            
                self.sums[name] += value * batch_size
                self.counts[name] += batch_size
            
    def get_average(self)-> Dict[str, float]:
        """Get averages as CPU float values"""
        with torch.no_grad():
            return {
                name: (self.sums[name] / self.counts[name]).cpu().item()
                for name in self.sums
            }
        
