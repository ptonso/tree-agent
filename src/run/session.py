import os
import random
import numpy as np
import torch
from typing import List

from src.env.env import LabEnvironment
from src.run.logger import create_logger
from src.agent.agent import Agent
from src.agent.trajectory import Trajectory
from src.agent.structures import Observation, State, Action

from src.explain.soft_tree import SoftDecisionTree, SoftConfig
from src.explain.visual.visualizer import Visualizer

class Session:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.seed = config.session.seed
        self.n_steps = config.session.n_steps

        self.online_trajectories = []
        self.replay_trajectories = []

        self.visualizer = None
        self.dtree = None
        self.agent = None

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

            if self.config.session.vae_vis and step % 400 == 0:

                with torch.no_grad():
                    x_hat = self.agent.world_model.decode(state)
                    # saliency = self.agent.world_model.find_objects(observation.as_tensor)[0, 0].cpu().numpy()
                    self.visualizer.update(
                        observation=observation.for_render,
                        state=state,
                        decoded=x_hat.for_render,
                        world_model=self.agent.world_model,
                    #     saliency=saliency,
                        tree=self.dtree
                    )
                    self.visualizer.render()

                    savepath = self.config.session.vis_prints_path
                    if episode_id > self.config.session.dtree_warmup_episodes \
                        and savepath is not None:
                        self.visualizer.save(savepath)

            trajectory.append(
                state            = state.state_data,
                action           = action.sampled_action,
                action_prob      = action.action_probs,
                reward           = reward,
                done             = done,
                next_state       = next_state.state_data,
                observation      = observation.obs_data,
                next_observation = next_observation.obs_data,
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
        dtree_metrics = {}

        for episode in range(self.config.session.n_episodes):
            episode_reward, trajectory = self.run_episode(episode)

            import cv2
            cv2.waitKey(1)

            total_rewards.append(episode_reward)
            self.online_trajectories.append(trajectory)

            self.logger.info("-"*35)
            self.logger.info(f"Episode {episode + 1:03d}: Total Reward = {episode_reward}")
            self.logger.info("-"*35)

            if len(self.online_trajectories) >= self.config.session.online_buffer:
                wm_train_metrics = self.agent.world_model.train_step(self.online_trajectories)
                self.transfer_to_replay(self.online_trajectories)
                self.online_trajectories = []
                wm_metrics[episode] = wm_train_metrics

            if len(self.replay_trajectories) >= self.config.session.replay_buffer:
                agent_train_metrics = {}
                dtree_train_metrics = {}
                if episode >= self.config.session.vae_warmup_episodes:
                    agent_train_metrics = self.agent.train_step(self.replay_trajectories)
                    
                    if episode >= self.config.session.dtree_warmup_episodes:
                        dtree_train_metrics = self.dtree.train_step(self.replay_trajectories)    

                agent_metrics[episode] = agent_train_metrics
                dtree_metrics[episode] = dtree_train_metrics
                self.replay_trajectories = []
                
        self.env.close()
        return total_rewards, wm_metrics, agent_metrics, dtree_metrics

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

        action_dim = self.config.agent.action_dim
        state_dim = self.config.agent.world_model.latent_dim
        # H, W, C = observation.shape
        # state_dim = (C, H, W)

        self.agent = Agent(action_dim=action_dim, state_dim=state_dim, config=self.config, logger=self.logger)

        self.dtree = SoftDecisionTree(config=self.config.soft, logger=self.logger)

        self.visualizer = Visualizer(window_name="Visualizer")

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