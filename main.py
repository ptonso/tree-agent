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
    baseline_config.session.n_episodes = 500

    parametric_changes = [
        {
            "exp": {"name": "small_nets_large_lr"},
            "agent": {"actor_lr": 5e-4, "critic_lr": 5e-4, "actor_layers": [32, 32], "critic_layers": [32, 32]},
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "low_res"},
            "agent": {},
            "env": {"width": 48, "height": 36},  # 16:12 * 3
            "session": {},
        },
        {
            "exp": {"name": "high_res"},
            "agent": {},
            "env": {"width": 96, "height": 72}, # 16:12 * 6
            "session": {},
        },
        {
            "exp": {"name": "high_exploration"},
            "agent": {"entropy_coef": 0.15},
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "lower_gamma"},
            "agent": {"gamma": 0.95},
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "large_nets"},
            "agent": {"actor_lr": 3e-4, "critic_lr": 3e-4, "actor_layers": [128, 128, 64], "critic_layers": [128, 128]},
            "env": {},
            "session": {},
        },
    ]

    base_results_dir = "experiments/mvp"
    multi_experiment = MultiExperiment(baseline_config, parametric_changes, results_dir=base_results_dir)
    multi_experiment.run_experiments()
    

if __name__ == "__main__":
    main()