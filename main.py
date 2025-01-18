from src.session import Session
from src.config import Config, SessionConfig, EnvConfig, AgentConfig


def main():
    config = Config(
        session=SessionConfig(
            name="baseline",
            n_episodes=200,
            n_steps=1000,
            seed=42,
            render=True,
        ),
        env=EnvConfig(
            width=80,# 80,
            height=60,# 60,
            fps=60,
            level="seekavoid_arena_01",
            num_steps=4,
            render_width=780,
            render_height=480
        ),
        agent=AgentConfig(
            actor_lr=0.0005,
            critic_lr=0.001,
            cnn_lr=0.0005,
            gamma=0.99,
            entropy_coef=0.01,
            actor_layers=[64, 64],
            critic_layers=[128],
        )
    )

    session = Session(config)
    session.setup()
    session.run()


if __name__ == "__main__":
    main()