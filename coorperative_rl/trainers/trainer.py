from coorperative_rl.agents.qtable_agent import QTableAgent
from coorperative_rl.envs import Environment
from coorperative_rl.states import AgentType
from coorperative_rl.agents.qtable import QTable


def train_qtable_based_agents(n: int = 4, num_episodes: int = 300) -> None:
    # agents share the same q-value matrix because the agents are symmetric
    qval_matrix = QTable(n=n)
    agents = [
        QTableAgent(agent_id=0, agent_type=AgentType.TYPE_A, qval_matrix=qval_matrix),
        QTableAgent(agent_id=1, agent_type=AgentType.TYPE_A, qval_matrix=qval_matrix),
        QTableAgent(agent_id=2, agent_type=AgentType.TYPE_B, qval_matrix=qval_matrix),
        QTableAgent(agent_id=3, agent_type=AgentType.TYPE_B, qval_matrix=qval_matrix),
    ]

    env = Environment(grid_size=n)
    for agent in agents:
        env.add_agent(agent)

    # TODO: implement the training loop
    for episode in range(num_episodes):
        env.initialize_for_new_episode(agent_states=None, goal_location=None)
        is_done = False
        while not is_done:
            env.start_new_step()

            for agent in agents:
                action = agent.decide_action(
                    possible_actions=env.get_available_actions(agent),
                    state=env.state.get_observable_state_for_agent(agent),
                )
                env.step_one_agent(agent, action)

            previous_state, current_state, rewards, is_done = env.end_step()

            for agent in agents:
                agent.update_model(
                    original_state=previous_state[agent],
                    moved_state=current_state[agent],
                    reward=rewards[agent],
                    action=env.action_taken[agent],
                )
