from typing import Literal

import fire

from coorperative_rl.agents.qtable_agent import QTableAgent
from coorperative_rl.envs import Environment
from coorperative_rl.states import AgentType
from coorperative_rl.training.core import validate
from coorperative_rl.utils import load_models


# use same params as traiing
def evaluate(
    model_checkpoint_path: str,
    num_samples: int | Literal["all"] = 1000,
    grid_size: int = 5,
    # env params
    goal_state_reward: int | float = 50,
    key_share_reward: int | float = 100,
    goal_without_key_penalty: int | float = -100,
    time_penalty: int | float = -1,
    visualize_env_train: bool = False,
) -> dict[str, float]:
    # NOTE: currently, this only supports two agents of each type
    models = load_models(model_checkpoint_path)

    agents = [
        QTableAgent(
            agent_id=0,
            agent_type=AgentType.TYPE_A,
            qval_matrix=models[0],
            epsilon_initial=None,
            epsilon_final=None,
            alpha=None,
            discount_rate=None,
        ),
        QTableAgent(
            agent_id=1,
            agent_type=AgentType.TYPE_A,
            qval_matrix=models[1],
            epsilon_initial=None,
            epsilon_final=None,
            alpha=None,
            discount_rate=None,
        ),
        QTableAgent(
            agent_id=2,
            agent_type=AgentType.TYPE_B,
            qval_matrix=models[2],
            epsilon_initial=None,
            epsilon_final=None,
            alpha=None,
            discount_rate=None,
        ),
        QTableAgent(
            agent_id=3,
            agent_type=AgentType.TYPE_B,
            qval_matrix=models[3],
            epsilon_initial=None,
            epsilon_final=None,
            alpha=None,
            discount_rate=None,
        ),
    ]

    env = Environment(
        grid_size=grid_size,
        visualize=visualize_env_train,
        goal_state_reward=goal_state_reward,
        key_share_reward=key_share_reward,
        goal_without_key_penalty=goal_without_key_penalty,
        time_penalty=time_penalty,
    )
    for agent in agents:
        env.add_agent(agent)

    newest_validation_metrics = validate(
        agents,
        env,
        tracker=None,
        num_samples=num_samples,
        with_progress_bar=True,
    )
    metric_names = [
        "average reward",
        "average path length",
        "goal reached percentage",
        "less than 15 steps percentage",
        "average excess path length",
    ]
    return {
        metric_name: metric_value
        for metric_name, metric_value in zip(metric_names, newest_validation_metrics)
    }


if __name__ == "__main__":
    fire.Fire(evaluate)
