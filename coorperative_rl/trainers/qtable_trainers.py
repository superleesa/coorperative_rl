from tqdm import tqdm

from coorperative_rl.agents.qtable_agent import QTableAgent
from coorperative_rl.envs import Environment
from coorperative_rl.states import AgentType
from coorperative_rl.agents.qtable import QTable


def train_qtable_based_agents(n: int = 4, num_episodes: int = 300, visualize: bool = True) -> None:
    # agents share the same q-value matrix because the agents are symmetric
    qval_matrix = QTable(n=n)
    agents = [
        QTableAgent(agent_id=0, agent_type=AgentType.TYPE_A, qval_matrix=qval_matrix),
        QTableAgent(agent_id=1, agent_type=AgentType.TYPE_A, qval_matrix=qval_matrix),
        QTableAgent(agent_id=2, agent_type=AgentType.TYPE_B, qval_matrix=qval_matrix),
        QTableAgent(agent_id=3, agent_type=AgentType.TYPE_B, qval_matrix=qval_matrix),
    ]

    env = Environment(grid_size=n, visualize=visualize)
    for agent in agents:
        env.add_agent(agent)

    for episode in tqdm(range(num_episodes)):
        run_episode(agents, env)
