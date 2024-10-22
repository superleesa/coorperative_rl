import itertools
from functools import lru_cache
from typing import Sequence

from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import ObservableState, AgentType


def split_dict_by_agent_type(
    agent_states: dict[BaseAgent, ObservableState]
    | Sequence[tuple[BaseAgent, ObservableState]],
) -> dict[AgentType, dict[BaseAgent, ObservableState]]:
    type_to_agent_states: dict[AgentType, dict[BaseAgent, ObservableState]] = {}

    if isinstance(agent_states, dict):
        for agent, state in (
            agent_states.items() if isinstance(agent_states, dict) else agent_states
        ):
            if state.agent_type not in type_to_agent_states:
                type_to_agent_states[state.agent_type] = {}
            type_to_agent_states[state.agent_type] |= {agent: state}

    return type_to_agent_states


def calculate_time_in_one_axis(
    a: int, b: int, g: int, requires_move_around_goal: bool = False
) -> int:
    """
    Calculate the time required for two agents to meet in one axis.

    Args:
        a: agent 1 position of the axis
        b: agent 2 position of the axis
        g: goal position of the axis
        requires_move_around_goal: whether the agents need to move around the goal
            (i.e. do we need to realocate the meeting point if it's on the goal)

    Returns:
        int: the time required for the agents to meet in the given axis
    """
    assert a <= b

    # pt1: goal outside of agent pair
    if a >= g or b <= g:
        pos_closer_to_goal = a if a >= b else b
        pos_further_from_goal = b if a >= b else a
        time_from_agent_to_meeting_point_upper_bound = abs(
            pos_closer_to_goal - pos_further_from_goal
        )

        # if meeting point is on the goal, so we need to move around it
        if pos_closer_to_goal == g and requires_move_around_goal:
            time_from_agent_to_meeting_point_upper_bound += 1

    # pt2: goal between agent pair
    else:
        meeting_point = median = (
            a + b
        ) // 2  # if number of cells is even, we take the left one

        # move the meeting point to the right / above side if meeting point was on goal
        if meeting_point == g and requires_move_around_goal:
            meeting_point += 1

        time_from_agent_to_meeting_point_upper_bound = max(
            abs(b - median), abs(a - median)
        )

    time_from_meeting_point_to_goal = abs(
        meeting_point - g
    )  # this is the same for both agents
    total_time = (
        time_from_agent_to_meeting_point_upper_bound + time_from_meeting_point_to_goal
    )

    return total_time


def calcluate_optiaml_time_estimation_for_agent_pair(
    agent1_location: tuple[int, int],
    agent2_location: tuple[int, int],
    goal_location: tuple[int, int],
) -> int:
    """
    Calculate the optimal time (step) estimation to solve the key-sharing problem between **two** agents.
    i.e. the time required for two agents to:
    1) meet before reaching the goal to share the keym, and
    2) reach the goal.
    """
    xa, ya = agent1_location
    xb, yb = agent2_location
    xg, yg = goal_location

    # ensure: xa <= xb, ya <= yb
    if xa > xb:
        xa, xb = xb, xa

    if ya > yb:
        ya, yb = yb, ya

    # important invariance:
    # because we use manhattan distance,
    # we can calculate the time in x and y-axes independently
    # HOWEVER: if an agent needs to "move around" the goal, there is some dependency between the axes
    # this happends when the goal is between the agents both in x-axis and y-axis
    # (and thier meeting point is on the goal, which will be checked inside `calculate_time_in_one_axis`)
    requires_move_around_goal = (xa <= xg <= xb) and (ya <= yg <= yb)

    time_in_x_axis = calculate_time_in_one_axis(
        xa, xb, xg, requires_move_around_goal
    )  # we only need the shift in one axis so we choose x-axis (but can be y-axis as well)
    time_in_y_axis = calculate_time_in_one_axis(ya, yb, yg)

    total_time = time_in_x_axis + time_in_y_axis
    return total_time


@lru_cache(maxsize=5000)  # covers about 1/3 of 5*5 grid env with 2 agents
def calculate_optimal_time_estimation_cached(
    agent_states: tuple[tuple[BaseAgent, ObservableState], ...],
    goal_location: tuple[int, int],
) -> tuple[
    int, tuple[tuple[BaseAgent, ObservableState], tuple[BaseAgent, ObservableState]]
]:
    """
    Calculate the optimal time (step) estimation to solve the key-sharing problem between any number of agent (but currently it only support 2 agent types).
    """
    type_to_agent_states = split_dict_by_agent_type(agent_states)
    type_to_agent_states_tuples = [
        [(agent, state) for agent, state in _agent_states.items()]
        for _agent_states in type_to_agent_states.values()
    ]  # we convert to tuple to avoid mypy error
    all_possible_agent_combinations = itertools.product(*type_to_agent_states_tuples)

    shortest_time = float("inf")
    shortest_time_agent_combination = None

    for agent_combination in all_possible_agent_combinations:  # should only be 8 if 4 agents with 2 types
        if len(agent_combination) != 2:
            raise ValueError("only support two agent types for now")

        agent1, agent2 = agent_combination
        time = calcluate_optiaml_time_estimation_for_agent_pair(
            agent1[1].agent_location, agent2[1].agent_location, goal_location
        )
        if time < shortest_time:
            shortest_time = time
            shortest_time_agent_combination = (agent1, agent2)

    if not isinstance(shortest_time, int) or shortest_time_agent_combination is None:
        raise ValueError("no optimal time found")

    return shortest_time, shortest_time_agent_combination


def calculate_optimal_time_estimation(
    agent_states: dict[BaseAgent, ObservableState], goal_location: tuple[int, int]
) -> tuple[
    int, tuple[tuple[BaseAgent, ObservableState], tuple[BaseAgent, ObservableState]]
]:
    """
    Calculate the optimal time (step) estimation to solve the key-sharing problem between any number of agent (but currently it only support 2 agent types).
    """
    _agent_states = tuple(agent_states.items())  # we need to make it hashable for caching
    return calculate_optimal_time_estimation_cached(_agent_states, goal_location)
