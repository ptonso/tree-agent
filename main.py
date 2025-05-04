
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
            n_episodes=1000,
            n_steps=1000,
            seed=42,
            render=False, # True
            vae_vis=True,
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
            name="better-dtree",
            n_runs=3
        ),
        session=SessionConfig(
            n_episodes=1000,
            n_steps=1000,
            seed=42,
            render=False,
            vae_vis=True,
        )
    )
    experiment = Experiment(config, base_dir="experiments")
    experiment.run_experiment()
    experiment.plot_results(savefig=True)
    experiment.plot_dtree_results(savefig=True)


def multi_exp_main():
    baseline_config = Config()
    baseline_config.session.render = False
    baseline_config.session.vae_vis = True
    baseline_config.session.n_episodes = 500
    baseline_config.session.dtree_warmup_episodes = 500

    parametric_changes = [
        {
            "exp": {"name": "bonus_free"},
            "agent": {
                "world_model": {
                    "saliency_bonus": 0.0,
                    "blur_kernel": 0
                }
            }
        },
        {
            "exp": {"name": "saliency_bonus"},
            "agent": {
                "world_model": {
                    "saliency_bonus": 3.0
                    }
                }
        },
    ]


    base_results_dir = "experiments/is-sobel-useful?"
    multi_experiment = MultiExperiment(baseline_config, parametric_changes, base_dir=base_results_dir)
    multi_experiment.run_experiments()
    

if __name__ == "__main__":
    main()