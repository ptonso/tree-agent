import os
import json
import time
import datetime
import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt

from src.run.session import Session
from src.run.config import Config, SessionConfig
from src.run.logger import create_logger

# classes for save:
from src.agent.agent import Agent
from src.agent.actor import Actor
from src.agent.critic import Critic

class Experiment:
    """Handles multiple training runs to test a configuration/hypothesis"""
    def __init__(self, config: Config, results_dir: str = 'experiments'):
        self.config = config
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        log_file = os.path.join(self.results_dir, f"{self.config.exp.name}_log.txt")
        self.logger = create_logger(log_file, logger_name=self.config.exp.name)
        

    def run_experiment(self, visualizer: bool = False) -> str:
        """Run multiple training sessions and collect results
        Args:
            visualizer: Whether to use visualization
            
        Returns:
            Filename where results are saved
        """
        all_rewards = []
        all_times = []

        for run_idx in range(self.config.exp.n_runs):
            self.logger.info(f"{'-'*30}\nStarting training run {run_idx + 1}/{self.config.exp.n_runs}")
            t0 = time.time()
            
            self.config.session.seed = self.config.exp.seed + run_idx
            
            session = Session(self.config, self.logger)
            session.setup()
            rewards = session.run()
            run_time = time.time() - t0

            self.logger.info("Average rewards:")
            self.logger.info(f"{'initial':10} {'middle':10} {'final':10}")
            division = len(rewards) // 3
            self.logger.info(f"{np.mean(rewards[:division]):<10.4f} "
                f"{np.mean(rewards[division:-division]):<10.4f} "
                f"{np.mean(rewards[-division:]):<10.4f}")
            self.logger.info(f"Training time: {run_time:.2f}s")

            all_rewards.append(rewards)
            all_times.append(run_time)

        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        
        experiment_data = {}

        experiment_data["config"] = {
            "exp": {field: getattr(self.config.exp, field)
                for field in self.config.exp.__dataclass_fields__},
            "session": {field: getattr(self.config.session, field)
                for field in self.config.session.__dataclass_fields__},
            "env": {field: getattr(self.config.env, field)
                for field in self.config.env.__dataclass_fields__},
            "agent": {field: getattr(self.config.agent, field)
                for field in self.config.agent.__dataclass_fields__},
            }
        
        experiment_data["code"] = {
            "agent": inspect.getsource(Agent),
            "actor": inspect.getsource(Actor),
            "critic": inspect.getsource(Critic)
            }
        
        experiment_data["performance"] = {
            'avg_rewards': avg_rewards.tolist(),
            'std_rewards': std_rewards.tolist(),
            'training_times': all_times,
            'mean_training_time': np.mean(all_times),
            'std_training_time': np.std(all_times)
        }
        self.experiment_data = experiment_data
        self.save_results(experiment_data)

    def save_results(self, experiment_data: dict) -> None:
        """Save experiment results to JSON file"""
        self.filename = f"{self.config.exp.name}_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.results_dir, self.filename)
        
        with open(filepath, 'w') as f:
            json.dump(experiment_data, f, indent=4)
        self.logger.info(f"Results saved to {filepath}")

    def plot_results(self, savefig: bool = True) -> None:
        """Plot results from one or more experiment files
        Args:
            files: List of result files to plot. If None, plots current experiment
        """
        plt.figure(figsize=(10, 6))

        performance = self.experiment_data['performance']
        avg_rewards = performance['avg_rewards']
        std_rewards = performance['std_rewards']
        
        episodes = np.arange(len(avg_rewards))
        plt.fill_between(
            episodes,
            np.array(avg_rewards) - np.array(std_rewards),
            np.array(avg_rewards) + np.array(std_rewards),
            alpha=0.2
        )
        plt.plot(episodes, avg_rewards, label=self.config.exp.name)

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if savefig:
            fig_path = os.path.join(self.results_dir, f"{self.config.exp.name}.png")    
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.5)

