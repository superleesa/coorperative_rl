import itertools
import time
from copy import deepcopy
from typing import Sequence
import random

from tqdm import tqdm

from coorperative_rl.actions import Action
from coorperative_rl.envs import Environment
from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import ObservableState, AgentState

from coorperative_rl.utils import generate_grid_location_list, generate_random_location


def run_episode(
    agents: Sequence[BaseAgent],
    env: Environment,
    kill_episode_after: float | int = 10,
    env_episode_initialization_params: dict | None = None,
) -> list[
    tuple[
        dict[BaseAgent, ObservableState],
        dict[BaseAgent, Action],
        dict[BaseAgent, float],
        dict[BaseAgent, ObservableState],
    ]
]:
    """
    A generic function that runs a single episode for a list of agents in a given environment.

    Args:
        agents: A list of agents participating in the episode.
        env: The environment in which the episode takes place.
        kill_episode_after: Maximum duration (in seconds) for the episode. Defaults to 10.
        env_episode_initialization_params: Parameters for initializing the environment for the episode. Defaults to None.
    Returns: A list of tuples containing the state-action-reward-state (SARS) history for the episode.
    """
    sars_history: list[
        tuple[
            dict[BaseAgent, ObservableState],
            dict[BaseAgent, Action],
            dict[BaseAgent, float],
            dict[BaseAgent, ObservableState],
        ]
    ] = []  # FIXME: maybe this has a huge memory footprint

    env_episode_initialization_params = (
        env_episode_initialization_params
        if env_episode_initialization_params is not None
        else dict(agent_states=None, goal_location=None, allow_overlapping_objects=True)
    )
    env.initialize_for_new_episode(env_episode_initialization_params)

    is_done = False
    start_time = time.time()
    while not is_done:
        if time.time() - start_time > kill_episode_after:
            break

        env.start_new_step()

        for agent in agents:
            action = agent.decide_action(
                possible_actions=env.get_available_actions(agent),
                state=env.state.get_observable_state_for_agent(agent),
            )
            env.step_one_agent(agent, action)

        previous_state, current_state, rewards, is_done = env.end_step()
        sars_history.append((previous_state, env.action_taken, rewards, current_state))

        for agent in agents:
            agent.update_model(
                original_state=previous_state[agent],
                moved_state=current_state[agent],
                reward=rewards[agent],
                action=env.action_taken[agent],
            )

    return sars_history


