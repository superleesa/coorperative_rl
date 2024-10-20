import fire

from coorperative_rl.trainers.qtable_trainers import train_qtable_based_agents

def main() -> None:
    train_qtable_based_agents()

if __name__ == "__main__":
    fire.Fire(main)