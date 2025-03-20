import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from src.agent.structures import State, Observation
from src.agent.vae import Encoder, Decoder
from src.agent.transition_model import TransitionModel
from src.agent.mlp import MLP
from src.agent.agent import Actor
from src.agent.batch import Batch, BatchTensors

class WorldModel(nn.Module):
    def __init__(self, config: object, actor: Actor, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.config = config
        self.logger = logger
        self.actor = actor
        self.device = config.device
        self._init_hyperparameter(config.agent.world_model)
        self._init_networks(config)

        self.saliency_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval().to(self.device)
        self.saliency_cache = None

    def _init_hyperparameter(self, wm_cfg):
        self.state_dim = wm_cfg.latent_dim
        self.beta_pred = wm_cfg.beta_pred
        self.beta_dym = wm_cfg.beta_dym
        self.beta_rep = wm_cfg.beta_rep
        self.gamma = self.config.agent.gamma
        self.horizon = wm_cfg.horizon
        self.free_nats = wm_cfg.free_nats

        self.blur_kernel = wm_cfg.blur_kernel
        self.saliency_bonus = wm_cfg.saliency_bonus

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

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.horizon > 0:
            parameters += (
                list(self.reward_predictor.parameters()) +
                list(self.continue_predictor.parameters()) +
                list(self.transition_model.parameters())
            )

        self.wm_optimizer = torch.optim.Adam(
            parameters,
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

        
    def find_objects(self, observation: torch.Tensor) -> torch.Tensor:
        """sobel-based edge detection"""
        with torch.no_grad():

            grayscale = 0.2989 * observation[:, 0, :, :] + \
                        0.5870 * observation[:, 1, :, :] + \
                        0.1140 * observation[:, 2, :, :]
            grayscale = grayscale.unsqueeze(1) # [B, 1, H, W]

            sobel_x = torch.tensor([[[[-1,  0,  1],
                                      [-2,  0,  2],
                                      [-1,  0,  1]]]], dtype=torch.float, device=observation.device)

            sobel_y = torch.tensor([[[[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2,  1]]]], dtype=torch.float, device=observation.device)


            grad_x = F.conv2d(grayscale, sobel_x, padding=1)
            grad_y = F.conv2d(grayscale, sobel_y, padding=1)

            edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            
            edge_magnitude = (edge_mag - edge_mag.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                             (edge_mag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

            
            # if is even, increment by 1
            blur_kernel_size = self.blur_kernel + 1 if self.blur_kernel % 2 == 0 else self.blur_kernel

            blur_weight = torch.ones((1, 1, blur_kernel_size, blur_kernel_size), device=edge_magnitude.device) / (blur_kernel_size ** 2)
            edge_magnitude = F.conv2d(edge_magnitude, blur_weight, padding=blur_kernel_size//2)
            
            return edge_magnitude # [B, 1, H, W] values in [0..1]


    def compute_losses(self, mb_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        1) Encode + predict the entire sequence of length T+1.
        2) Reconstruction losses for all frames.
        3) Reward/Continue losses for frames 1..T if T>0.
        4) Dynamic & representation KL for frames 1..T if T>0.
        """
        obs_seq      = mb_data['obs_seq']       # [b, H+1, C, H, W]
        action_seq   = mb_data['action_seq']    # [b, H]
        reward_seq   = mb_data['reward_seq']    # [b, H]
        continue_seq = mb_data['continue_seq']  # [b, H]
        b, Hplus1, C, H, W = obs_seq.shape
        h = Hplus1 - 1

        mu0, logvar0 = self.encoder(obs_seq[:, 0])
        z0 = self.reparametrize(mu0, logvar0)

        z_seq = [z0]
        mu_seq = [mu0]
        logvar_seq = [logvar0]
        r_preds = []
        c_preds = []

        for t in range(h):
            mu_hat, logvar_hat = self.transition_model(z_seq[-1], action_seq[:, t])
            z_next = self.reparametrize(mu_hat, logvar_hat)

            z_seq.append(z_next)
            mu_seq.append(mu_hat)
            logvar_seq.append(logvar_hat)
            
            r_preds.append(self.reward_predictor(z_next))
            c_preds.append(self.continue_predictor(z_next))

        z_seq = torch.stack(z_seq, dim=1)
        mu_seq = torch.stack(mu_seq, dim=1)
        logvar_seq = torch.stack(logvar_seq, dim=1)

        reward_hat = torch.stack(r_preds, dim=1) if h > 0 else torch.zeros(b, 0, 1, device=self.device)
        continue_hat = torch.stack(c_preds, dim=1) if h > 0 else torch.zeros(b, 0, 1, device=self.device)
        
        z_flat = z_seq.view(b * (h+1), -1)
        obs_hat = self.decoder(z_flat)

        obs_seq_flat = obs_seq.view(b * (h+1), C, H, W)
        symlog_obs_seq = self.symlog(obs_seq_flat)

        saliency = self.find_objects(obs_seq_flat)
        bonus_weight = 1.0 + self.saliency_bonus * saliency
        recon_loss = (F.mse_loss(obs_hat, symlog_obs_seq, reduction='none') * bonus_weight).mean() / bonus_weight.mean()

        if h > 0:
            symlog_r = self.symlog(reward_seq.unsqueeze(-1))
            reward_loss = F.mse_loss(reward_hat, symlog_r, reduction='mean')
            cont_loss = F.binary_cross_entropy_with_logits(continue_hat, continue_seq.unsqueeze(-1), reduction='mean')

            mu_p = mu_seq[:, 1:].reshape(b * h, -1)
            logvar_p = logvar_seq[:, 1:].reshape(b * h, -1)

            with torch.no_grad():
                mu_q, logvar_q = self.encoder(obs_seq[:, 1:].reshape(b * h, C, H, W))

            # KL[ stopgrad(q) || p ]
            dyn_loss = self.gaussian_kl(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p)

            # KL[ q || stopgrad(p) ]
            rep_loss = self.gaussian_kl(mu_p.detach(), logvar_p.detach(), mu_q, logvar_q)
            
            kl_adj = self.kl_scale * (dyn_loss.detach() - self.kl_ema)
            self.kl_ema = 0.99 * self.kl_ema + 0.01 * dyn_loss.detach()
        else:
            reward_loss, cont_loss, dyn_loss, rep_loss, kl_adj = [torch.tensor(0.0, device=self.device)] * 5

        total_loss = (
            self.beta_pred * (recon_loss + reward_loss + cont_loss)
            + (self.beta_dym + kl_adj) * dyn_loss
            + self.beta_rep * rep_loss
        )

        return {
            'total_loss':    total_loss,
            'recon_loss':    self.beta_pred * recon_loss,
            'reward_loss':   self.beta_pred * reward_loss,
            'continue_loss': self.beta_pred * cont_loss,
            'dyn_loss':      self.beta_dym * dyn_loss,
            'rep_loss':      self.beta_rep * rep_loss,
        }




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
            trajectories: List['Trajectory']
        ):
        """Train world model using list of trajectory"""
        mb_size = self.config.agent.world_model.mb_size
        n_epochs = self.config.agent.world_model.n_epochs
        horizon = self.config.agent.world_model.horizon
        
        tensors = Batch(trajectories, self.device).prepare_tensors()

        metrics_accumulator = MetricAccumulator(self.device, self.horizon)

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
        self._log_metrics(final_metrics)
    
        return final_metrics
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics if loger is provided"""
        if self.logger:
            self.logger.info(f"World Model Loss:    {metrics['total_loss']:.4f}")
            self.logger.info(f"    Recon Loss:      {metrics['recon_loss']:.4f}")
            self.logger.info(f"    Reward Loss:     {metrics['reward_loss']:.4f}")
            self.logger.info(f"    Continue Loss:   {metrics['continue_loss']:.4f}")
            self.logger.info(f"    Dyn Loss:        {metrics['dyn_loss']:.4f}")
            self.logger.info(f"    Rep Loss:        {metrics['rep_loss']:.4f}")


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
        Extract minibatch of sequeneces with shape [b, H+1, *dim]
        b : minibatch length
        H : horizon
        """
        b = len(indices)
        h = self.horizon
        seq_indices = indices.unsqueeze(1) + torch.arange(h + 1, device=self.device).unsqueeze(0)
        flat_seq = seq_indices.flatten()

        obs_seq = tensors.observations[flat_seq].view(b, h+1, *tensors.observations.shape[1:])

        if h == 0:
            action_seq = torch.zeros(b, 0, device=self.device)
            reward_seq = torch.zeros(b, 0, device=self.device)
            continue_seq= torch.zeros(b, 0, device=self.device)
        else:
            seq_act = indices.unsqueeze(1) + torch.arange(h, device=self.device).unsqueeze(0)
            flat_act = seq_act.flatten()

            action_seq   = tensors.actions[flat_act].view(b, self.horizon)
            reward_seq   = tensors.rewards[flat_act].view(b, self.horizon)
            continue_seq = tensors.dones[flat_act].view(b, self.horizon)


        return {
            'obs_seq': obs_seq,
            'action_seq': action_seq,
            'reward_seq': reward_seq,
            'continue_seq': continue_seq,
        }


class MetricAccumulator:
    """Helper class to accumulate and average training metrics"""
    def __init__(self, device: str, horizon: int):
            self.sums = {}
            self.counts = {}
            self.horizon = horizon
            self.device = device

    def update(self, losses: Dict[str, torch.Tensor], batch_size: int):
        """Update metrics maintaining tensors on device"""
        total_steps = batch_size * (self.horizon + 1)
        with torch.no_grad():
            for name, value in losses.items():
                if name not in self.sums:
                    self.sums[name] = torch.zeros(1, device=self.device)
                    self.counts[name] = 0
            
                self.sums[name] += value * total_steps
                self.counts[name] += total_steps
            
    def get_average(self)-> Dict[str, float]:
        """Get averages as CPU float values"""
        with torch.no_grad():
            return {
                name: (self.sums[name] / self.counts[name]).cpu().item()
                for name in self.sums
            }
        
