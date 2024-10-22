import itertools
import time
from copy import deepcopy
from typing import Sequence, TypeAlias, TypedDict

from tqdm import tqdm

from coorperative_rl.actions import Action
from coorperative_rl.envs import Environment
from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import ObservableState, AgentState
from coorperative_rl.metrics import calculate_optimal_time_estimation

from coorperative_rl.utils import (
    generate_grid_location_list,
    generate_random_location,
    shuffle_and_distribute_agents,
)
from coorperative_rl.trackers import BaseTracker


# TODO: port these types somewhere else
SARS: TypeAlias = tuple[
    dict[BaseAgent, ObservableState],
    dict[BaseAgent, Action],
    dict[BaseAgent, float],
    dict[BaseAgent, ObservableState],
]


class EpisodeSampleParams(TypedDict):
    agent_states: dict[BaseAgent, AgentState]
    goal_location: tuple[int, int]


def run_episode(
    agents: Sequence[BaseAgent],
    env: Environment,
    kill_episode_after: float | int = 10,
    env_episode_initialization_params: dict | None = None,
) -> tuple[list[SARS], bool]:
    """
    A generic function that runs a single episode for a list of agents in a given environment.

    Args:
        agents: A list of agents participating in the episode.
        env: The environment in which the episode takes place.
        kill_episode_after: Maximum duration (in seconds) for the episode. Defaults to 10.
        env_episode_initialization_params: Parameters for initializing the environment for the episode. Defaults to None.
    Returns: A list of tuples containing the state-action-reward-state (SARS) history for the episode. And has_reached_goal flag.
    """
    sars_history: list[SARS] = []  # FIXME: maybe this has a huge memory footprint

    env_episode_initialization_params = (
        env_episode_initialization_params
        if env_episode_initialization_params is not None
        else dict(agent_states=None, goal_location=None, allow_overlapping_objects=True)
    )
    env.initialize_for_new_episode(env_episode_initialization_params)

    has_reached_goal = True
    is_done = False
    start_time = time.time()
    while not is_done:
        if time.time() - start_time > kill_episode_after:
            has_reached_goal = False
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

    return sars_history, has_reached_goal


def generate_episode_samples(
    grid_size: int, agents: list[BaseAgent]
) -> list[EpisodeSampleParams]:
    # NOTE: [positions of agents to control] + [positions of agents that doesn't need to be controlled (so we randomly shuffle)]
    # [fixed agent position parts] + [unfixed agent positions parts]

    # consider all agent positions pairs (between n types of agents)
    # actual agent / goals will be randomly sampled
    num_types = len(set([agent.type for agent in agents]))
    num_total_samples = (grid_size * grid_size) ** num_types
    fixed_agent_position_parts = list(
        itertools.product(
            *[
                generate_grid_location_list(grid_size, grid_size)
                for _ in range(num_types)
            ]
        )
    )
    unfixed_agent_position_parts = [
        [generate_random_location(grid_size) for _ in range(len(agents) - 2)]
        for _ in range(num_total_samples)
    ]

    # concat above two lists
    agent_states_samples = [
        {
            agent: AgentState(
                id=agent.id, type=agent.type, location=location, has_full_key=False
            )
            for agent, location in zip(
                shuffle_and_distribute_agents(agents),
                list(fixed_agent_positions) + unfixed_agent_positions,
            )
        }
        for fixed_agent_positions, unfixed_agent_positions in zip(
            fixed_agent_position_parts, unfixed_agent_position_parts
        )
    ]

    goal_location_samples = [
        generate_random_location(grid_size) for _ in range(num_total_samples)
    ]
    episode_samples = [
        EpisodeSampleParams(agent_states=agent_states, goal_location=goal_location)
        for agent_states, goal_location in zip(
            agent_states_samples, goal_location_samples
        )
    ]
    return episode_samples


def validate(
    agents: list[BaseAgent],
    env: Environment,
    tracker: BaseTracker | None,
    validation_index: int | None = None,
) -> tuple[float, float, float, float, float]:
    """
    FIXME: maybe we need support more statistics
    This assumes that there is no difference in models between agents with the same type (to reduce number of possible start states).
    # TODO: implement sampling functinality
    """
    if tracker and validation_index is None:
        raise ValueError("validation_index must be provided when tracker is provided")

    env = deepcopy(env)
    episode_samples = generate_episode_samples(grid_size=env.grid_size, agents=agents)

    average_reward = 0.0
    average_path_length = 0.0
    goal_reached_percentage = 0.0
    less_than_15_steps_percentage = 0.0
    average_excess_path_length = 0.0
    for i, episode_sample in tqdm(enumerate(episode_samples)):
        sars_collected, has_reached_goal = run_episode(
            agents,
            env,
            env_episode_initialization_params=episode_sample,
            kill_episode_after=0.01,
        )

        episode_wise_average_reward = sum(
            [sum(sars[2].values()) for sars in sars_collected]
        ) / len(agents)
        average_reward += episode_wise_average_reward / len(episode_samples)
        episode_path_length = len(sars_collected)
        average_path_length += episode_path_length / len(episode_samples)
        goal_reached_percentage += has_reached_goal / len(episode_samples)
        less_than_15_steps_percentage += (has_reached_goal and episode_path_length < 15) / len(episode_samples)
        optimal_path_length, _ = calculate_optimal_time_estimation(episode_sample["agent_states"], episode_sample["goal_location"])
        average_excess_path_length += (episode_path_length - optimal_path_length) / len(episode_samples)

    if tracker is not None and validation_index is not None:
        tracker.log_metric(
            "validation_average_reward", average_reward, validation_index
        )
        tracker.log_metric(
            "validation_average_path_length", average_path_length, validation_index
        )
        tracker.log_metric(
            "validation_goal_reached_percentage",
            goal_reached_percentage,
            validation_index,
        )
        tracker.log_metric(
            "validation_less_than_15_steps_percentage",
            less_than_15_steps_percentage,
            validation_index,
        )
        tracker.log_metric(
            "validation_average_excess_path_length",
            average_excess_path_length,
            validation_index,
        )

    return average_reward, average_path_length, goal_reached_percentage, less_than_15_steps_percentage, average_excess_path_length


def visualize_samples(
    agents: list[BaseAgent], env: Environment, num_visualizations: int = 5
) -> None:
    env = deepcopy(env)
    env.visualizer.visualize = True

    episode_samples = [
        {
            "agent_states": {
                agent: AgentState(
                    id=agent.id,
                    type=agent.type,
                    location=generate_random_location(env.grid_size),
                    has_full_key=False,
                )
                for agent in agents
            },
            "goal_location": generate_random_location(env.grid_size),
        }
        for _ in range(num_visualizations)
    ]

    for episode_sample in episode_samples:
        run_episode(agents, env, env_episode_initialization_params=episode_sample)

    env.visualizer.close()
