
from src.run.session import Session
from src.run.experiment import Experiment
from src.run.multi_exp import MultiExperiment
from src.run.config import Config, SessionConfig, EnvConfig, ExpConfig


def main():
    rendered_main()
    # single_exp_main()
    # multi_exp_main()
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
            name="dtree-gaussian-rebuild",
            n_runs=3
        ),
        session=SessionConfig(
            n_episodes=500,
            n_steps=1000,
            seed=42,
            render=False,
            vae_vis=False,
        )
    )
    experiment = Experiment(config, base_dir="experiments")
    experiment.run_experiment()
    experiment.plot_results(savefig=True)
    experiment.plot_dtree_results(savefig=True)


def multi_exp_main():
    baseline_config = Config()
    baseline_config.session.render = False
    baseline_config.session.vae_vis = False
    baseline_config.session.n_episodes = 500

    parametric_changes = [
        {
            "exp": {"name": "dtree20conc1sharpen"},
            "soft": {"concentration": 20.0,
                    "sharpen":1.0}
        },
        {
            "exp": {"name": "dtree30concs1harpen"},
            "soft": {"concentration": 30.0,
                    "sharpen":1.0}
        },
        {
            "exp": {"name": "dtree10conc1sharpen"},
            "soft": {"concentration": 20.0,
                    "sharpen":1.0}
        },
        {
            "exp": {"name": "dtree20conc2sharpen"},
            "soft": {"concentration": 20.0,
                    "sharpen":2.0}
        },
        {
            "exp": {"name": "dtree20conc1.5sharpen"},
            "soft": {"concentration": 20.0,
                    "sharpen":1.5}
        },
    ]


    base_results_dir = "experiments/dtree-full-rebuild"
    multi_experiment = MultiExperiment(baseline_config, parametric_changes, base_dir=base_results_dir)
    multi_experiment.run_experiments()
    

if __name__ == "__main__":
    main()