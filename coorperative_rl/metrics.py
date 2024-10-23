import itertools

from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import AgentState, AgentType


def split_by_type(
    agent_states: tuple[
        tuple[
            int,
            AgentType,
            tuple[int, int],
        ],
        ...,
    ],
) -> tuple[tuple[tuple[int, tuple[int, int]], ...], ...]:
    type_to_agent_states: dict[AgentType, dict[int, tuple[int, int]]] = {}

    for agent_id, agent_type, agent_location in agent_states:
        if agent_type not in type_to_agent_states:
            type_to_agent_states[agent_type] = {}
        type_to_agent_states[agent_type][agent_id] = agent_location

    return tuple(
        [
            tuple(
                [
                    (agent_id, agent_location)
                    for agent_id, agent_location in agent_states.items()
                ]
            )
            for agent_states in type_to_agent_states.values()
        ]
    )


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
    # --> meet at where the closer agent (to the goal) is
    # (technically, the meeting point can be anywhere between the goal and b)
    if a >= g or b <= g:
        meeting_point = pos_closer_to_goal = a if abs(a - g) <= abs(b - g) else b
        pos_further_from_goal = b if abs(a - g) <= abs(b - g) else a
        time_from_agent_to_meeting_point_upper_bound = abs(
            pos_closer_to_goal - pos_further_from_goal
        )

        # if meeting point is on the goal, we need to move around it
        if pos_closer_to_goal == g and requires_move_around_goal:
            time_from_agent_to_meeting_point_upper_bound += 1
            meeting_point = pos_closer_to_goal + 1 if a >= g else pos_closer_to_goal - 1

    # pt2: goal between agent pair
    # --> meet at the median of the agent pair
    else:
        has_two_options = (a + b) % 2 == 0

        median_left = (a + b) // 2
        meeting_point_candiates = (
            [median_left, median_left + 1] if has_two_options else [median_left]
        )

        # it's better to take the one at goal (if any of the agents is on the goal)
        # but if requires_move_around_goal, we must not be on the goal
        if any([candiate == g for candiate in meeting_point_candiates]):
            if requires_move_around_goal:
                filtered_candiates = [
                    candiates for candiates in meeting_point_candiates if candiates != g
                ]
                if not filtered_candiates:
                    meeting_point = (
                        meeting_point_candiates[0] + 1
                    )  # FIXME: this might go over the grid?
                else:
                    meeting_point = filtered_candiates[0]
            else:
                filtered_candiates = [
                    candiates for candiates in meeting_point_candiates if candiates == g
                ]
                meeting_point = filtered_candiates[0]

        else:
            # general case: it doesn't matter if it's not on the goal
            meeting_point = meeting_point_candiates[0]

        time_from_agent_to_meeting_point_upper_bound = max(
            abs(b - meeting_point), abs(a - meeting_point)
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


def _calculate_optimal_time_estimation(
    agent_states: tuple[
        tuple[
            int,
            AgentType,
            tuple[int, int],
        ],
        ...,
    ],
    goal_location: tuple[int, int],
) -> tuple[int, tuple[int, ...]]:
    """
    Calculate the optimal time (step) estimation to solve the key-sharing problem between any number of agent (but currently it only support 2 agent types).
    """
    type_to_agent_states = split_by_type(agent_states)

    all_possible_agent_combinations = itertools.product(*type_to_agent_states)

    shortest_time = float("inf")
    shortest_time_agent_combination = None

    for (
        agent_combination
    ) in all_possible_agent_combinations:  # should only be 8 if 4 agents with 2 types
        if len(agent_combination) != 2:
            raise ValueError("only support two agent types for now")

        (agent1_id, agent1_location), (agent2_id, agent2_location) = agent_combination

        time = calcluate_optiaml_time_estimation_for_agent_pair(
            agent1_location, agent2_location, goal_location
        )
        if time < shortest_time:
            shortest_time = time
            shortest_time_agent_combination = (agent1_id, agent2_id)

    if not isinstance(shortest_time, int) or shortest_time_agent_combination is None:
        raise ValueError("no optimal time found")

    return shortest_time, shortest_time_agent_combination


def calculate_optimal_time_estimation(
    agent_states: dict[BaseAgent, AgentState],
    goal_location: tuple[int, int],
) -> tuple[int, tuple[BaseAgent, ...]]:
    """
    Calculate the optimal time (step) estimation to solve
    the key-sharing problem between any number of agent (but currently it only support 2 agent types).

    This is only a wrapper function for `calculate_optimal_time_estimation_cached`
    that converts the input to hashable format with minimum information.
    """
    # extract the minimum information required to avoide cache miss
    _agent_states = tuple(
        [
            (agent.id, agent.type, agent_state.location)
            for agent, agent_state in agent_states.items()
        ]
    )

    shortest_time, shortest_time_agent_combination = _calculate_optimal_time_estimation(
        _agent_states, goal_location
    )

    agent_id_to_agent = {agent.id: agent for agent in agent_states.keys()}
    agent_combination = tuple(
        [agent_id_to_agent[agent_id] for agent_id in shortest_time_agent_combination]
    )

    return shortest_time, agent_combination
