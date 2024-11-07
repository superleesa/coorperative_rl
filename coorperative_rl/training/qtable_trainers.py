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
    model_sharing_level: Literal[
        "shared-all", "shared-type", "separate"
    ] = "shared-type",
    use_central_clock: bool = True,
    track: bool = True,
    tracker_backend: str = "mlflow",
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
    """
    A tabular Q-learning trainer for cooperative agents.
    
    Args:
        grid_size: The size of the grid world
        num_episodes: The number of episodes to train the agents
        model_sharing_level: The level of sharing the Q-value matrix between agents. Options are "shared-all", "shared-type", and "separate".
        use_central_clock: Flag indicating whether to use a central clock for the agents. Defaults to True.
        off_the_job_training: Flag indicating whether to use off-the-job training. Defaults to False. Note that if this is set to True, number of episodes will be ignored.
        track: Flag indicating whether to track the training process. Defaults to True.
        tracker_backend: The type of tracker to use. Defaults to "mlflow".
        validation_interval: The interval at which to validate the agents. Defaults to 10.
        visualization_env_validation_interval: The interval at which to visualize the environment during validation. Defaults to 100.
        do_final_evaluation: Flag indicating whether to do final evaluation after training. Defaults to True.
        epsilon_initial: The initial epsilon value for epsilon-greedy action selection. Defaults to 0.9.
        epsilon_final: The final epsilon value for epsilon-greedy action selection. Defaults to 0.1.
        alpha: The learning rate for updating the Q-value matrix. Defaults to 0.3.
        discount_rate: The discount rate for future rewards. Defaults to 0.9.
        goal_state_reward: The reward for reaching the goal state. Defaults to 50.
        key_share_reward: The reward for sharing the key. Defaults to 100.
        goal_without_key_penalty: The penalty for reaching the goal state without the key. Defaults to -100.
        time_penalty: The penalty for each time step. Defaults to -1.
        initialization_has_full_key_prob: The probability of having the full key at the start of the episode. Defaults to 0.0.
        visualize_env_train: Flag indicating whether to visualize the environment during training. Defaults to True.
    
    Returns:
        The validation metrics (see validate function) and the Q-value matrix models.
    """
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

    tracker = select_tracker(track, tracker_backend)

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
                agents,
                env,
                is_training=True,
                use_central_clock=use_central_clock,
                kill_episode_after=0.05,
                env_episode_initialization_params={
                    "has_full_key_prob": initialization_has_full_key_prob
                },
            )

            if (
                validation_interval is not None
                and episode_idx % validation_interval == 0
            ):
                newest_validation_metrics = validate(
                    agents,
                    env,
                    tracker,
                    use_central_clock=use_central_clock,
                    validation_index=episode_idx,
                )

            if (
                visualization_env_validation_interval is not None
                and episode_idx % visualization_env_validation_interval == 0
            ):
                visualize_samples(agents, env)

            # update epsilon
            for agent in agents:
                agent.update_hyper_parameters(episode_idx, num_episodes)

        if do_final_evaluation:
            newest_validation_metrics = validate(
                agents,
                env,
                tracker,
                use_central_clock=use_central_clock,
                validation_index=num_episodes,
            )

    return newest_validation_metrics, models
