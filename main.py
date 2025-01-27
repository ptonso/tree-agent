from src.run.session import Session
from src.run.experiment import Experiment
from src.run.multi_exp import MultiExperiment
from src.run.config import Config, SessionConfig, EnvConfig, ExpConfig


def main():
    # rendered_main()
    # single_exp_main()
    multi_exp_main()
    pass


def rendered_main():
    print("[simple render] Starting session run")
    config = Config(
        session=SessionConfig(
            n_episodes=200,
            n_steps=1000,
            seed=42,
            render=True,
        ),
    env=EnvConfig(
            width=80,
            height=60
        )
    )
    session = Session(config)
    session.setup()
    session.run()
    print("[simple_render] Done.")



def single_exp_main():
    config = Config(
        exp=ExpConfig(
            name="mvp",
            n_runs=10
        ),
        session=SessionConfig(
            n_episodes=100,
            n_steps=1000,
            seed=42,
            render=True
        )
    )
    experiment = Experiment(config, results_dir="experiments/mvp")
    experiment.run_experiment()
    experiment.plot_results(savefig=True)


def multi_exp_main():
    baseline_config = Config()
    baseline_config.session.render = False

    parametric_changes = [
        {
            "exp": {"name": "small_nets_large_lr"},
            "session": {"seed": 1, "n_episodes": 10},
            "agent": {"actor_lr": 5e-4, "critic_lr": 5e-4, "actor_layers": [32, 32], "critic_layers": [32, 32]},
            "env": {}
        },
        {
            "exp": {"name": "low_res"},
            "session": {"seed": 1, "n_episodes": 10},
            "agent": {},
            "env": {"width": 48, "height": 36}  # 16:12 * 3
        },
        {
            "exp": {"name": "high_res"},
            "session": {"seed": 1, "n_episodes": 10},
            "agent": {},
            "env": {"width": 96, "height": 72} # 16:12 * 6
        },
        {
            "exp": {"name": "high_exploration"},
            "session": {"seed": 1},
            "agent": {"entropy_coef": 0.15},
            "env": {}
        },
        {
            "exp": {"name": "lower_gamma"},
            "session": {"seed": 1},
            "agent": {"gamma": 0.95},
            "env": {}
        },
        {
            "exp": {"name": "large_nets"},
            "session": {"seed": 1},
            "agent": {"actor_lr": 3e-4, "critic_lr": 3e-4, "actor_layers": [128, 128, 64], "critic_layers": [128, 128]},
            "env": {}
        },
    ]

    base_results_dir = "experiments/mvp"
    multi_experiment = MultiExperiment(baseline_config, parametric_changes, results_dir=base_results_dir)
    multi_experiment.run_experiments()
    

if __name__ == "__main__":
    main()