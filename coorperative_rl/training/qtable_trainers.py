from typing import Literal

from tqdm import tqdm

from coorperative_rl.agents.qtable_agent import QTableAgent
from coorperative_rl.envs import Environment
from coorperative_rl.states import AgentType
from coorperative_rl.agents.qtable import QTable
from coorperative_rl.training.core import run_episode, validate, visualize_samples
from coorperative_rl.trackers import select_tracker


def train_qtable_based_agents(
    grid_size: int = 5,
    num_episodes: int = 300,
    model_sharing_level: Literal["shared-all", "shared-type", "separate"] = "shared-type",
    track: bool = True,
    tracker_type: str = "mlflow",
    validation_interval: int | None = 10,
    visualization_env_validation_interval: int | None = 100,
    do_final_evaluation: bool = True,
    # agent params
    epsilon_initial: float = 0.9,
    epsilon_final: float = 0.1,
    alpha: float = 0.3,
    discount_rate: float = 0.9,
    # env params
    goal_state_reward: int | float = 50,
    key_share_reward: int | float = 100,
    goal_without_key_penalty: int | float = -100,
    time_penalty: int | float = -1,
    initialization_has_full_key_prob: float = 0.0,
    visualize_env_train: bool = True,
) -> tuple[tuple[float, float, float, float, float] | None, list[QTable]]:
    # NOTE: currently, this only supports two agents of each type
    
    # NOTE: agents can share the same q-value matrix if the agents are symmetric
    if model_sharing_level == "shared-all":
        shared_model = QTable(n=grid_size)
        models = [shared_model for _ in range(4)]
    elif model_sharing_level == "shared-type":
        shared_model_a = QTable(n=grid_size)
        shared_model_b = QTable(n=grid_size)
        models = [shared_model_a, shared_model_a, shared_model_b, shared_model_b]
    else:
        models = [QTable(n=grid_size) for _ in range(4)]
    
    agents = [
        QTableAgent(
            agent_id=0,
            agent_type=AgentType.TYPE_A,
            qval_matrix=models[0],
            epsilon_initial=epsilon_initial,
            epsilon_final=epsilon_final,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
        QTableAgent(
            agent_id=1,
            agent_type=AgentType.TYPE_A,
            qval_matrix=models[1],
            epsilon_initial=epsilon_initial,
            epsilon_final=epsilon_final,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
        QTableAgent(
            agent_id=2,
            agent_type=AgentType.TYPE_B,
            qval_matrix=models[2],
            epsilon_initial=epsilon_initial,
            epsilon_final=epsilon_final,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
        QTableAgent(
            agent_id=3,
            agent_type=AgentType.TYPE_B,
            qval_matrix=models[3],
            epsilon_initial=epsilon_initial,
            epsilon_final=epsilon_final,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
    ]

    tracker = select_tracker(track, tracker_type)

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

    newest_validation_metrics = None
    with tracker:
        for episode_idx in tqdm(range(num_episodes)):
            run_episode(
                agents, env, is_training=True, kill_episode_after=0.05, env_episode_initialization_params={"has_full_key_prob": initialization_has_full_key_prob}
            )

            if (
                validation_interval is not None
                and episode_idx % validation_interval == 0
            ):
                newest_validation_metrics = validate(agents, env, tracker, validation_index=episode_idx)

            if (
                visualization_env_validation_interval is not None
                and episode_idx % visualization_env_validation_interval == 0
            ):
                visualize_samples(agents, env)
            
            # update epsilon
            for agent in agents:
                agent.update_hyper_parameters(episode_idx, num_episodes)
    
    if do_final_evaluation:
        newest_validation_metrics = validate(agents, env, tracker, validation_index=num_episodes)
    
    return newest_validation_metrics, models
