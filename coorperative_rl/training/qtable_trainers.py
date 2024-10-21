from tqdm import tqdm

from coorperative_rl.agents.qtable_agent import QTableAgent
from coorperative_rl.envs import Environment
from coorperative_rl.states import AgentType
from coorperative_rl.agents.qtable import QTable
from coorperative_rl.trainers.core import run_episode, validate, visualize_samples
from coorperative_rl.trackers import select_tracker


def train_qtable_based_agents(
    n: int = 4,
    num_episodes: int = 300,
    visualize: bool = True,
    track: bool = True,
    tracker_type: str = "mlflow",
    validation_interval: int = 10,
    visualization_interval: int = 100,
) -> None:
    # agents share the same q-value matrix because the agents are symmetric
    qval_matrix = QTable(n=n)
    agents = [
        QTableAgent(agent_id=0, agent_type=AgentType.TYPE_A, qval_matrix=qval_matrix),
        QTableAgent(agent_id=1, agent_type=AgentType.TYPE_A, qval_matrix=qval_matrix),
        QTableAgent(agent_id=2, agent_type=AgentType.TYPE_B, qval_matrix=qval_matrix),
        QTableAgent(agent_id=3, agent_type=AgentType.TYPE_B, qval_matrix=qval_matrix),
    ]

    tracker = select_tracker(track, tracker_type)

    env = Environment(grid_size=n, visualize=visualize)
    for agent in agents:
        env.add_agent(agent)

    with tracker:
        for episode in tqdm(range(num_episodes)):
            run_episode(
                agents, env
            )  # for now, we don't plot statistics for training loops

            if episode % validation_interval == 0:
                validate(agents, tracker, episode)

            if episode % visualization_interval == 0:
                visualize_samples(agents, env)
