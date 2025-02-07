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
            name="test!!",
            n_runs=3
        ),
        session=SessionConfig(
            n_episodes=500,
            n_steps=1000,
            seed=42,
            render=False,
            vae_vis=True,
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
            "exp": {"name": "reward_loss"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_reward": 0.2}
            },
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "return_loss"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_return": 0.2}
            },
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "triplet_loss"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_triplet": 0.2}
            },
            "env": {}, 
            "session": {},
        },
        {
            "exp": {"name": "high_beta_kl"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_kl": 0.1},
            },
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "agressive_reward_based"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {"beta_reward": 0.3, "beta_returns": 0.3, "beta_triplet": 0.3},
            },
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "large_actor_critic"},
            "agent": {
                "actor": {"layers": [128, 128, 64]},
                "critic": {"layers": [128, 128]},
                "world_model": {}
            },
            "env": {},
            "session": {},
        },
        {
            "exp": {"name": "more_vae_fc"},
            "agent": {
                "actor": {},
                "critic": {},
                "world_model": {
                    "encoder": {"fc_layers": 3},
                    "decoder": {"fc_layers": 3}
                }
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