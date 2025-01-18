import random
import numpy as np
import torch

from typing import List

from .env import LabEnvironment
from .agent import Agent
from .trajectory import Trajectory
from .structures import State

class Session:
    def __init__(self, config):
        self.config = config
        self.seed = config.session.seed
        self.n_steps = config.session.n_steps

    def run_episode(self, episode_id: int = 0) -> float:
        """Run single episode"""
        self.trajectory = Trajectory(device=self.config.device, trajectory_id=episode_id, gamma=self.config.agent.gamma)
        observation = self.env.reset(seed=self.seed + episode_id)

        episode_reward = 0.
        done = False

        state = State(observation, device=self.config.device)

        for step in range(self.n_steps):
            action = self.agent.policy(state)
            next_observation, reward, done = self.env.step(action.as_lab)
            next_state = State(next_observation, device=self.config.device)
            episode_reward += reward

            if self.config.session.render:
                self.env.render(next_observation)

            self.trajectory.append(
                state.state_data,
                action.sampled_action,
                reward,
                done,
                next_state.state_data
            )

            if done:
                observation = self.env.reset()
                state = State(observation, device=self.config.device)
            else:
                state = next_state
            
            if self.visualizer:
                self.visualize(state, action)
        
        return episode_reward
    

    def run(self, visualizer: bool = False) -> List[float]:
        """Run multiple episodes collecting experience"""
        total_rewards = []
        self.visualizer = visualizer
        for episode in range(self.config.session.n_episodes):
            episode_reward = self.run_episode(episode)
            total_rewards.append(episode_reward)

            print(f"\n Episode {episode + 1:03d}: Total Reward = {episode_reward}")

            trajectories = [self.trajectory]
            self.agent.train(trajectories)

        self.env.close()
        return total_rewards


    def setup(self):
        """Initialize environment and agent"""
        self._set_seed(self.seed)

        self.env = LabEnvironment(config=self.config)
        observation = self.env.reset(seed=self.seed)

        H, W, C = observation.shape
        state_dim = H * W * C
        action_dim = self.env.action_space

        self.agent = Agent(
            state_dim,
            action_dim,
            config=self.config
        )


    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False