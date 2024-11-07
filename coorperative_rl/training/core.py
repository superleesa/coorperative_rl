import itertools
import random
import time
from copy import deepcopy
from typing import Sequence, Literal

from tqdm import tqdm

from coorperative_rl.envs import Environment
from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import AgentState
from coorperative_rl.metrics import calculate_optimal_time_estimation

from coorperative_rl.utils import (
    batched,
    generate_grid_location_list,
    generate_random_location,
    shuffle_and_distribute_agents,
    split_agent_list_by_type,
)
from coorperative_rl.trackers import BaseTracker
from coorperative_rl.types import EpisodeSampleParams, SARS


def run_episode(
    agents: Sequence[BaseAgent],
    env: Environment,
    is_training: bool,
    use_central_clock: bool = True,
    kill_episode_after: float | int = 10,
    env_episode_initialization_params: EpisodeSampleParams | dict | None = None,
) -> tuple[list[SARS], bool]:
    """
    Runs a single episode for a list of agents in a given environment.

    Args:
        agents: A list of agents participating in the episode.
        env: The environment in which the episode takes place.
        is_training: A flag indicating whether the episode is for training purposes. It won't update agent model states if set to False.
        use_central_clock: A flag indicating whether to use a central clock for the episode (i.e. train agents in order or randomly). Defaults to True.
        kill_episode_after: Maximum duration (in seconds) for the episode. Defaults to 10.
        env_episode_initialization_params: Parameters for initializing the environment for the episode. Defaults to None.

    Returns:
        A tuple containing:
            - A list of tuples representing the state-action-reward-state (SARS) history for the episode.
            - A boolean flag indicating whether the goal was reached.
    """
    sars_history: list[SARS] = []  # FIXME: maybe this has a huge memory footprint

    env_episode_initialization_params = (
        env_episode_initialization_params
        if env_episode_initialization_params is not None
        else dict(agent_states=None, goal_location=None, allow_overlapping_objects=True)
    )
    env.initialize_for_new_episode(**env_episode_initialization_params)

    has_reached_goal = True
    is_done = False
    start_time = time.time()
    while not is_done:
        if time.time() - start_time > kill_episode_after:
            has_reached_goal = False
            break

        env.start_new_step()

        if not use_central_clock:
            agents = random.sample(agents, len(agents))
        
        for agent in agents:
            action = agent.decide_action(
                possible_actions=env.get_available_actions(agent),
                state=env.state.get_observable_state_for_agent(agent),
                is_training=is_training,
            )
            env.step_one_agent(agent, action)

        previous_state, current_state, rewards, is_done = env.end_step()
        sars_history.append((previous_state, env.action_taken, rewards, current_state))

        if not is_training:
            continue

        for agent in agents:
            agent.update_model(
                original_state=previous_state[agent],
                moved_state=current_state[agent],
                reward=rewards[agent],
                action=env.action_taken[agent],
            )

    return sars_history, has_reached_goal


def generate_episode_samples_all_agent_pair_locations(
    grid_size: int, agents: list[BaseAgent]
) -> list[EpisodeSampleParams]:
    # FIXME: weired function name
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


def generate_episode_samples(
    grid_size: int, agents: list[BaseAgent], num_samples: int
) -> list[EpisodeSampleParams]:
    """
    Generate episode samples for the given agents and grid size.
    
    Args:
        grid_size: The size of the grid.
        agents: A list of agents participating in the episode.
        num_samples: Number of samples to generate.
    
    Returns:
        A list of episode samples.
    """
    return [
        {
            "agent_states": {
                agent: AgentState(
                    id=agent.id,
                    type=agent.type,
                    location=generate_random_location(grid_size),
                    has_full_key=False,
                )
                for agent in agents
            },
            "goal_location": generate_random_location(grid_size),
        }
        for _ in range(num_samples)
    ]


def generate_full_episode_samples(
    grid_size: int, agents: list[BaseAgent]
) -> tuple[list[EpisodeSampleParams], list[BaseAgent]]:
    """
    Generate all possible episode samples for the given agents and grid size.
    Use it if you want to validate on all possible environments (this will take a huge amount of time for larger env).
    Note that this function would only consider one agent from each type (to reduce the number of possible start states).
    
    Args:
        grid_size: The size of the grid.
        agents: A list of agents participating in the episode.
    
    Returns:
        A tuple containing:
            - A list of all possible episode samples.
            - A list of agents that are used within the samples.
    """
    location_indices = [location for location in range(grid_size)]

    # pick one agent from each type (we don't need all agents because the game is symmetric)
    chosen_agents = [
        random.choice(agents_of_type)
        for agents_of_type in split_agent_list_by_type(agents).values()
    ]

    sample_params_raw = itertools.product(
        location_indices,
        location_indices,
        *[
            location_indices for _ in range(len(chosen_agents) * 2)
        ],  # *2 because each agent has a location pair (x, y)
    )

    return (
        [
            EpisodeSampleParams(
                agent_states={
                    chosen_agent: AgentState(
                        id=0,
                        type=chosen_agent.agent_type,
                        location=agent_location,
                        has_full_key=False,
                    )
                    for chosen_agent, agent_location in zip(
                        chosen_agents,
                        batched(
                            sample_param_raw[2:], 2
                        ),  # chunk of size 2 because each location is a pair (x, y)
                    )
                },
                goal_location=sample_param_raw[:2],  # type: ignore
            )
            for sample_param_raw in sample_params_raw
        ],
        chosen_agents,
    )


def validate(
    agents: Sequence[BaseAgent],
    env: Environment,
    tracker: BaseTracker | None,
    num_samples: int | Literal["all"] = 1000,
    use_central_clock: bool = True,
    validation_index: int | None = None,
    with_progress_bar: bool = False,
    episode_wise_logger: BaseTracker |  None = None,
) -> tuple[float, float, float, float, float]:
    """
    Validate the performance of agents in a given environment.
    
    Args:
        agents: A list of agents participating in the episode.
        env: The environment in which the episode takes place.
        tracker: A tracker to log the metrics. (This is to log the average metric values only once. If you want to log for each validation sample, use `episode_wise_logger`) If provided, `validation_index` must also be provided.
        num_samples: Number of samples to generate for validation. Defaults to 1000. Set to "all" to validate on all possible environments (this will take a huge amount of time for larger environments).
        use_central_clock: A flag indicating whether to use a central clock for the episode (i.e. train agents in order or randomly). Defaults to True.
        validation_index: The index of the validation. Used for logging. Defaults to None.
        with_progress_bar: Whether to display a progress bar during validation. Defaults to False.
        episode_wise_logger: A tracker to log episode-wise metrics. Defaults to None.
    
    Returns:
        tuple:
        - average_reward: The average reward obtained by the agents.
        - average_path_length: The average path length taken by the agents.
        - goal_reached_percentage: The percentage of episodes where the goal was reached.
        - less_than_15_steps_percentage: The percentage of episodes where the goal was reached in less than 15 steps.
        - average_excess_path_length: The average excess path length over the optimal path length for episodes where the goal was reached.
    """
    if tracker and validation_index is None:
        raise ValueError("validation_index must be provided when tracker is provided")

    env = deepcopy(env)
    if num_samples == "all":
        episode_samples, chosen_agents = generate_full_episode_samples(
            grid_size=env.grid_size, agents=agents
        )
        # remove agents that are not used within the samples
        unused_agents = [agent for agent in agents if agent not in chosen_agents]
        for agent in unused_agents:
            env.remove_agent(agent)

        agents = chosen_agents

    elif isinstance(num_samples, int):
        episode_samples = generate_episode_samples(
            grid_size=env.grid_size, agents=agents, num_samples=num_samples
        )  # 1/3 of the total possible states
    else:
        raise ValueError("num_samples must be an integer or 'all'")

    average_reward = 0.0
    average_path_length = 0.0
    goal_reached_percentage = 0.0
    less_than_15_steps_percentage = 0.0
    average_excess_path_length_sum = 0.0
    num_has_reached_goal = 0
    for sample_episode_idx, episode_sample in enumerate(
        tqdm(episode_samples, disable=not with_progress_bar)
    ):
        sars_collected, has_reached_goal = run_episode(
            agents,
            env,
            is_training=False,
            use_central_clock=use_central_clock,
            env_episode_initialization_params=episode_sample,
            kill_episode_after=0.01,
        )
        episode_path_length = len(sars_collected)

        episode_wise_average_reward = sum(
            [sum(sars[2].values()) for sars in sars_collected]
        ) / len(agents)
        optimal_path_length, _ = calculate_optimal_time_estimation(
            episode_sample["agent_states"], episode_sample["goal_location"]
        )
        episode_wise_excess_path_length = (episode_path_length - optimal_path_length) if has_reached_goal else 0
        
        if episode_wise_logger is not None:
            episode_wise_logger.log_metric(
                "episode_reward", episode_wise_average_reward, sample_episode_idx
            )
            episode_wise_logger.log_metric(
                "episode_path_length", episode_path_length, sample_episode_idx
            )
            episode_wise_logger.log_metric(
                "episode_has_reaced_goal", has_reached_goal, sample_episode_idx
            )
            episode_wise_logger.log_metric(
                "episode_excess_path_length", episode_wise_excess_path_length, sample_episode_idx
            )
            if episode_wise_logger.can_log_sars():
                episode_wise_logger.log_sars(sars_collected, sample_episode_idx)
        
        num_has_reached_goal += has_reached_goal
        average_reward += episode_wise_average_reward / len(episode_samples)
        average_path_length += episode_path_length / len(episode_samples)
        goal_reached_percentage += has_reached_goal / len(episode_samples)
        less_than_15_steps_percentage += (
            has_reached_goal and episode_path_length < 15
        ) / len(episode_samples)
        average_excess_path_length_sum += episode_wise_excess_path_length

    # for average excess path length, we only consider the cases where the goal is reached
    average_excess_path_length = average_excess_path_length_sum / num_has_reached_goal

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

    return (
        average_reward,
        average_path_length,
        goal_reached_percentage,
        less_than_15_steps_percentage,
        average_excess_path_length,
    )


def visualize_samples(
    agents: Sequence[BaseAgent], env: Environment, num_visualizations: int = 5
) -> None:
    """
    Visualize the samples of episodes for the given agents and environment.
    
    Args:
        agents: A list of agents participating in the episode.
        env: The environment in which the episode takes place.
        num_visualizations: Number of visualizations to generate. Defaults to 5.
    """
    env = deepcopy(env)
    env.visualizer.visualize = True

    episode_samples = generate_episode_samples(
        grid_size=env.grid_size, agents=agents, num_samples=num_visualizations
    )

    for episode_sample in episode_samples:
        run_episode(
            agents,
            env,
            is_training=False,
            env_episode_initialization_params=episode_sample,
        )
    
    env.visualizer.close()


class CoordinatedEnvParameterIterator:
    def __iter__(self):
        return self

    def __next__(self) -> dict:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError


class EmptyParamIterator(CoordinatedEnvParameterIterator):
    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def __next__(self) -> dict:
        if self.current_iteration >= self.max_iterations:
            raise StopIteration
        
        self.current_iteration += 1
        return {}
    
    def __len__(self) -> int:
        return self.max_iterations


class CoordinatedGoalLocationIterator(CoordinatedEnvParameterIterator):
    def __init__(self, grid_size: int = 5, num_episodes_per_combination: int = 800, num_cycles: int = 5) -> None:
        """
        Args:
            grid_size (int): The size of the grid (e.g., 5 for a 5x5 grid).
            num_episodes_per_combination (int): The number of episodes to run for each goal location combination. Defaults to 800.
            num_cycles (int): If num_cycles > 1, the iterator will cycle through goal locations step-by-step, rather than exhausting one goal location before moving to the next. This can help in reducing catastrophic forgetting. Defaults to 5.
        """
        self.num_episodes_per_combination = num_episodes_per_combination
        self.current_episode_index = -1
        self.current_goal_combination_index = -1
        
        self.all_goal_locations = list(itertools.product([x for x in range(grid_size)], [y for y in range(grid_size)]))
        self.num_episodes_per_cycle = num_episodes_per_combination // num_cycles
    
    def __next__(self) -> dict:
        if self.current_episode_index >= self.num_episodes_per_combination * len(self.all_goal_locations):
            raise StopIteration
        
        if self.current_episode_index % self.num_episodes_per_cycle == 0:
            self.current_goal_combination_index = (self.current_goal_combination_index + 1) % len(self.all_goal_locations)
            
        self.current_episode_index += 1
        return  {"goal_location": self.all_goal_locations[self.current_goal_combination_index]}

    def __len__(self) -> int:
        return self.num_episodes_per_combination * len(self.all_goal_locations)
