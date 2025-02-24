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
            render=True,
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
            name="kl_div",
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


def multi_exp_main():
    baseline_config = Config()
    baseline_config.session.render = False
    baseline_config.session.vae_vis = False
    baseline_config.session.n_episodes = 500

    parametric_changes = [
        {
            "exp": {"name": "td_q_n3_loss"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_td_q": 0.2}
            },
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "td_q_n8_loss"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_td_q": 0.2, "n_steps": 8},
            },
            "env": {},
            "session": {},
        },
    ]

    base_results_dir = "experiments/vae"
    multi_experiment = MultiExperiment(baseline_config, parametric_changes, base_dir=base_results_dir)
    multi_experiment.run_experiments()
    

if __name__ == "__main__":
    main()