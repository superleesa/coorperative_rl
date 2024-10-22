import fire

from coorperative_rl.hp_tuning.hp_tuners import tune_qtable


def main(study_name: str) -> None:
    tune_qtable(study_name)


if __name__ == "__main__":
    fire.Fire(main)