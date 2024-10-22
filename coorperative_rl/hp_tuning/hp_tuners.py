import pickle
from typing import Callable

import optuna

from coorperative_rl.training.qtable_trainers import train_qtable_based_agents
from coorperative_rl.utils import generate_unique_id, save_checkpoint


def qtable_objective(trial: optuna.Trial) -> tuple[float, float, float, float, float]:
    training_id = generate_unique_id()
    trial.set_user_attr("training_id", training_id)

    computed_metrics, models = train_qtable_based_agents(
        track=False,
        validation_interval=None,
        visualization_env_validation_interval=None,
        visualize_env_train=False,
        num_episodes=trial.suggest_int("num_episodes", 500, 3000),
        alpha=trial.suggest_loguniform("alpha", 0.01, 0.9),  # ensure smaller learing rates are tried more
        discount_rate=trial.suggest_float("discount_rate", 0.7, 0.99),
        epsilon_initial=trial.suggest_float("epsilon_initial", 0.8, 1.0),
        epsilon_final=trial.suggest_float("epsilon_final", 0.01, 0.3),
        goal_state_reward=trial.suggest_int("goal_state", 10, 100),
        key_share_reward=trial.suggest_int("key_share_reward", 5, 100),
        goal_without_key_penalty=trial.suggest_int("goal_without_key_penalty", -100, -10),
        time_penalty=trial.suggest_int("time_penalty", -20, 0),
        initialization_has_full_key_prob=trial.suggest_float(
            "initialization_has_full_key_prob", 0.0, 1.0
        ),
    )

    # save the best model
    save_checkpoint(
        models,
        training_id,
    )
    
    if computed_metrics is None:
        raise ValueError("metrics must be calculated at least once within the objective function")

    return computed_metrics


def tune(study_name: str, objective: Callable, directions: list[str]) -> None:
    NUM_TRIALS = 100

    study = optuna.create_study(
        directions=directions,
        storage="sqlite:///tuning_result.db",
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)

    with open("best_trials.pickle", "wb") as file:
        pickle.dump(study.best_trials, file)


def tune_qtable(study_name: str) -> None:
    tune(study_name, qtable_objective, ["maximize", "minimize", "maximize", "maximize", "minimize"])