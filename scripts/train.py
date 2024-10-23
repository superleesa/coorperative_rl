import fire

from coorperative_rl.training.qtable_trainers import train_qtable_based_agents


def main() -> None:
    train_qtable_based_agents(
        num_episodes=300,
        track=False,
        # tracker_type="mlflow",
        visualize_env_train=False,
        visualization_env_validation_interval=10,
        validation_interval=None,
        do_final_evaluation=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
