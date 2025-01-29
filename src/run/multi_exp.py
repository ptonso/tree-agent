import os
import torch
from copy import deepcopy

from src.run.config import Config
from src.run.experiment import Experiment

from typing import List, Dict

class MultiExperiment:
    """Handles multiple experiments with different configurations and parallel execution"""
    def __init__(
            self, 
            baseline_config: Config, 
            parametric_changes: List[Dict],
            base_dir: str = 'multi-experiment'
            ):
        self.baseline_config = baseline_config
        self.parametric_changes = parametric_changes
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.experiment_configs = self.setup_experiments()

    def setup_experiments(self):
        experiment_configs = []
        for changes in self.parametric_changes:
            modified_config = deepcopy(self.baseline_config)
            MultiExperiment._apply_changes(modified_config, changes)
            experiment_configs.append(modified_config)
        return experiment_configs


    def run_experiments(self):
        for config in self.experiment_configs:
            print(f"Running experiment: {config.exp.name}")
            self._run_single_experiment(config, self.base_dir)
        input("Press ENTER to close plots")


        # num_workers = min(len(self.experiment_configs), self.max_processes)
        # with multiprocessing.Pool(processes=num_workers) as pool:
        #     pool.starmap(self._run_single_experiment, self.experiment_configs)

    @staticmethod
    def _apply_changes(config: Config, changes: Dict):
        "Apply a possibly nested dictionary of changes to configuration."
        for section, params in changes.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if isinstance(params, dict):
                    MultiExperiment._apply_changes(section_obj, params)
                else:
                    setattr(config, section, params)


    def _run_single_experiment(self, config: Config, base_dir: str):
        try:
            experiment = Experiment(config, base_dir=base_dir)
            experiment.run_experiment()    
            experiment.plot_results(savefig=True)
            print(f"[SUCCESS] Experiment {config.exp.name} run successfully.")
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] GPU out of memory in experiment: {config.exp.name}. Skipping...")
        finally:
            torch.cuda.empty_cache()


# added to venv/bin/activate script: 
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True