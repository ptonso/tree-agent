import os
import random
import numpy as np
import torch
from typing import List

from src.env.env import LabEnvironment
from src.run.logger import create_logger
from src.agent.agent import Agent
from src.agent.trajectory import Trajectory
from src.agent.structures import Observation, State

from src.explain.visualizer import AutoencoderVisualizer

class Session:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.seed = config.session.seed
        self.n_steps = config.session.n_steps

        self.online_trajectories = []
        self.replay_trajectories = []

        if logger is None:
            log_file = os.path.join("session_log.txt")
            self.logger = create_logger(log_file, logger_name="Session")


    def run_episode(self, episode_id: int = 0) -> float:
        """Run single episode"""
        trajectory = Trajectory(
            device=self.config.device, 
            trajectory_id=episode_id,
            gamma=self.config.agent.gamma,
            n_steps=3)
        
        obs_data = self.env.reset(seed=self.seed + episode_id)
        observation = Observation.from_env(obs_data, device=self.config.device)

        episode_reward = 0.
        done = False
        
        for step in range(self.n_steps):
            state = self.agent.world_model.encode(observation)

            action = self.agent.actor.policy(state)
            # action = self.agent.policy(observation)

            next_obs_data, reward, done = self.env.step(action.as_lab)
            next_observation = Observation.from_env(next_obs_data, device=self.config.device)
            next_state = self.agent.world_model.encode(next_observation)

            episode_reward += reward

            if self.config.session.render:
                self.env.render(observation.for_render, action)

            if self.config.session.vae_vis and step % 200 == 0:

                # saliency = self.agent.critic.compute_saliency(observation.as_tensor, next_observation.as_tensor)
                # saliency = Observation(saliency, device=self.config.device)

                with torch.no_grad():
                    x_hat = self.agent.world_model.decode(state)
                    self.vis.render(
                        [observation.for_render], 
                        [state.for_render], 
                        [x_hat.for_render], )
                        # [saliency.for_render])


            trajectory.append(
                state.state_data,
                action.sampled_action,
                reward,
                done,
                next_state.state_data,
                observation.obs_data,
                next_observation.obs_data,
            )

            if done:
                obs_data = self.env.reset()
                observation = Observation.from_env(obs_data, device=self.config.device)
            else:
                observation = next_observation

        return episode_reward, trajectory
    

    def run(self) -> List[float]:
        """Run multiple episodes collecting experience and training in batches"""
        total_rewards = []
        wm_metrics = {}
        agent_metrics = {}

        WORLD_MODEL_TRAJECTORIES_IN_BATCH = 1
        ACTOR_CRITIC_TRAJECTORIES_IN_BATCH = 4

        for episode in range(self.config.session.n_episodes):
            episode_reward, trajectory = self.run_episode(episode)
            total_rewards.append(episode_reward)
            self.online_trajectories.append(trajectory)

            self.logger.info("-"*35)
            self.logger.info(f"Episode {episode + 1:03d}: Total Reward = {episode_reward}")
            self.logger.info("-"*35)

            if len(self.online_trajectories) >= WORLD_MODEL_TRAJECTORIES_IN_BATCH:
                wm_train_metrics = self.agent.world_model.train_step(self.online_trajectories, logger=self.logger)
                self.transfer_to_replay(self.online_trajectories)
                self.online_trajectories = []
                wm_metrics[episode] = wm_train_metrics

            if len(self.replay_trajectories) >= ACTOR_CRITIC_TRAJECTORIES_IN_BATCH:
                agent_train_metrics = self.agent.train_step(self.replay_trajectories, logger=self.logger)
                self.replay_trajectories = []
                agent_metrics[episode] = agent_train_metrics

        self.env.close()
        return total_rewards, wm_metrics, agent_metrics

    def transfer_to_replay(self, trajectories: List[Trajectory]):
        """Convert online trajectories to replay buffer."""
        for traj in trajectories:
            traj.observations = None
            traj.next_observations = None
            self.replay_trajectories.append(traj)

    def setup(self):
        """Initialize environment and agent"""
        self._set_seed(self.seed)

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current_device)
            
            total_memory = props.total_memory / (1024 ** 2) # GB
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            
            self.logger.info(f"Using CUDA device: {current_device}")
            self.logger.info(f"Total GPU Memory:      {total_memory:6.2f} MB")
            self.logger.info(f"CUDA memory allocated: {mem_alloc:6.2f} MB")
            self.logger.info(f"CUDA memory reserved:  {mem_reserved:6.2f} MB")


        self.env = LabEnvironment(config=self.config)
        observation = self.env.reset(seed=self.seed)

        state_dim = self.config.agent.world_model.latent_dim
        # H, W, C = observation.shape
        # state_dim = (C, H, W)

        self.agent = Agent(action_dim=self.env.action_space, state_dim=state_dim, config=self.config)

        self.vis = AutoencoderVisualizer()

        self.logger.info("Session setup complete.")


    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.logger.info(f"Random seed set to {seed}")