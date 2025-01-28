import os
import random
import numpy as np
import torch

from typing import List

from src.env.env import LabEnvironment
from src.run.logger import create_logger
from src.agent.agent import Agent
from src.agent.trajectory import Trajectory
from src.agent.structures import State

class Session:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.seed = config.session.seed
        self.n_steps = config.session.n_steps

        if logger is None:
            log_file = os.path.join("session_log.txt")
            self.logger = create_logger(log_file, logger_name="Session")


    def run_episode(self, episode_id: int = 0) -> float:
        """Run single episode"""
        self.trajectory = Trajectory(device=self.config.device, trajectory_id=episode_id, gamma=self.config.agent.gamma)
        observation = self.env.reset(seed=self.seed + episode_id)

        episode_reward = 0.
        done = False
        
        state = State.from_observation(observation, device=self.config.device)

        for step in range(self.n_steps):
            action = self.agent.policy(state)
            next_observation, reward, done = self.env.step(action.as_lab)
            next_state = State.from_observation(next_observation, device=self.config.device)
            episode_reward += reward

            if self.config.session.render:
                self.env.render(next_observation, action)

            self.trajectory.append(
                state.state_data,
                action.sampled_action,
                reward,
                done,
                next_state.state_data
            )

            if done:
                observation = self.env.reset()
                state = State.from_observation(observation, device=self.config.device)
            else:
                state = next_state
                   
        return episode_reward
    

    def run(self, batch_size: int = 4) -> List[float]:
        """Run multiple episodes collecting experience and training in batches"""
        total_rewards = []
        trajectory_buffer = []

        for episode in range(self.config.session.n_episodes):
            episode_reward = self.run_episode(episode)
            total_rewards.append(episode_reward)
            trajectory_buffer.append(self.trajectory)

            self.logger.info(f"Episode {episode + 1:03d}: Total Reward = {episode_reward}")

            if (episode + 1) % batch_size == 0:
                self.logger.info(f"Training on batch of {len(trajectory_buffer)} trajectories")
                self.agent.train(trajectory_buffer, logger=self.logger)
                trajectory_buffer = []

        self.env.close()
        return total_rewards


    def setup(self):
        """Initialize environment and agent"""
        self._set_seed(self.seed)

        self.env = LabEnvironment(config=self.config)
        observation = self.env.reset(seed=self.seed)

        H, W, C = observation.shape
        state_dim = (C, H, W)
        action_dim = self.env.action_space

        self.agent = Agent(
            state_dim,
            action_dim,
            config=self.config
        )
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