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
    track: bool = True,
    tracker_type: str = "mlflow",
    validation_interval: int | None = 10,
    visualization_interval: int | None = 100,
    # agent params
    epsilon: float = 0.1,
    alpha: float = 0.3,
    discount_rate: float = 0.9,
    # env params
    goal_state_reward: int | float = 50,
    key_share_reward: int | float = 100,
    goal_without_key_penalty: int | float = -100,
    time_penalty: int | float = -1,
    visualize: bool = True,
) -> None:
    # agents share the same q-value matrix because the agents are symmetric
    qval_matrix = QTable(n=grid_size)
    agents = [
        QTableAgent(
            agent_id=0,
            agent_type=AgentType.TYPE_A,
            qval_matrix=qval_matrix,
            epsilon=epsilon,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
        QTableAgent(
            agent_id=1,
            agent_type=AgentType.TYPE_A,
            qval_matrix=qval_matrix,
            epsilon=epsilon,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
        QTableAgent(
            agent_id=2,
            agent_type=AgentType.TYPE_B,
            qval_matrix=qval_matrix,
            epsilon=epsilon,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
        QTableAgent(
            agent_id=3,
            agent_type=AgentType.TYPE_B,
            qval_matrix=qval_matrix,
            epsilon=epsilon,
            alpha=alpha,
            discount_rate=discount_rate,
        ),
    ]

    tracker = select_tracker(track, tracker_type)

    env = Environment(
        grid_size=grid_size,
        visualize=visualize,
        goal_state_reward=goal_state_reward,
        key_share_reward=key_share_reward,
        goal_without_key_penalty=goal_without_key_penalty,
        time_penalty=time_penalty,
    )
    for agent in agents:
        env.add_agent(agent)

    with tracker:
        for episode_idx in tqdm(range(num_episodes)):
            run_episode(
                agents, env
            )  # for now, we don't plot statistics for training loops

            if (
                validation_interval is not None
                and episode_idx % validation_interval == 0
            ):
                validate(agents, env, tracker, episode_idx)

            if (
                visualization_interval is not None
                and episode_idx % visualization_interval == 0
            ):
                visualize_samples(agents, env)
