import os
import json
import time
import datetime
import inspect
import numpy as np
import torch
from collections import defaultdict
from dataclasses import is_dataclass, fields
import matplotlib.pyplot as plt

from src.run.session import Session
from src.run.config import Config
from src.run.logger import create_logger

# classes for save:
from src.agent.agent import Agent
from src.agent.actor import Actor
from src.agent.critic import Critic
from src.agent.world_model import WorldModel


class Experiment:
    """Handles multiple training runs to test a configuration/hypothesis"""
    def __init__(self, config: Config, base_dir: str = "experiments"):
        self.config = config
        self.results_dir = os.path.join(base_dir, config.exp.name)
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
        wm_all_metrics = defaultdict(list)
        agent_all_metrics = defaultdict(list)
        dtree_all_metrics = defaultdict(list)

        for run_idx in range(self.config.exp.n_runs):
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Starting training run {run_idx + 1}/{self.config.exp.n_runs}")
            self.logger.info(f"{'='*50}")
            t0 = time.time()
            
            self.config.session.seed = self.config.seed + run_idx
            
            session = Session(self.config, self.logger)
            session.setup()
            rewards, wm_metrics, agent_metrics, dtree_metrics = session.run()
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

            for ep, metrics in wm_metrics.items():
                wm_all_metrics[ep+1].append(metrics)
            for ep, metrics in agent_metrics.items():
                agent_all_metrics[ep+1].append(metrics)
            for ep, metrics in dtree_metrics.items():
                dtree_all_metrics[ep+1].append(metrics)

        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        wm_avg, wm_std = self.aggregate_metrics(wm_all_metrics)
        agent_avg, agent_std = self.aggregate_metrics(agent_all_metrics)
        dtree_avg, dtree_std = self.aggregate_metrics(dtree_all_metrics)
       
        experiment_data = {
            "config": Experiment.dataclass2dict(self.config),
            "code": {
            "agent": inspect.getsource(Agent),
            "actor": inspect.getsource(Actor),
            "critic": inspect.getsource(Critic),
            "world_model": inspect.getsource(WorldModel),
            },
        "performance": {
            'avg_rewards': avg_rewards.tolist(),
            'std_rewards': std_rewards.tolist(),
            'training_times': all_times,
            'mean_training_time': np.mean(all_times),
            'std_training_time': np.std(all_times)
            },
        "train_metrics": {
            "world_model": {"avg": wm_avg, "std": wm_std},
            "agent": {"avg": agent_avg, "std": agent_std},
            "dtree": {"avg": dtree_avg, "std": dtree_std}

            }
        }

        self.experiment_data = experiment_data
        self.save_results(experiment_data)


    def aggregate_metrics(self, all_metrics):
        avg_metrics = {}
        std_metrics = {}

        for ep, metrics_list in all_metrics.items():
            metric_keys = metrics_list[0].keys()
            metrics_per_key = {key: [m.get(key, 0) for m in metrics_list] for key in metric_keys}

            avg_metrics[ep] = {key: float(np.mean(values)) for key, values in metrics_per_key.items()}
            std_metrics[ep] = {key: float(np.std(values)) for key, values in metrics_per_key.items()}

        return avg_metrics, std_metrics

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
            fig_path = os.path.join(self.results_dir, f"{self.filename}.png")    
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.5)

    @staticmethod
    def dataclass2dict(obj):
        if is_dataclass(obj):
            result = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = Experiment.dataclass2dict(value)
            return result
        elif isinstance(obj, list):
            return [Experiment.dataclass2dict(v) for v in obj]
        elif isinstance(obj, torch.nn.Module):
            return obj.__class__.__name__
        else:
            return obj