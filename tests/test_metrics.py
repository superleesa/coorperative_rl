import pytest
from coorperative_rl.metrics import (
    calculate_time_in_one_axis,
    calcluate_optiaml_time_estimation_for_agent_pair,
)


@pytest.mark.parametrize(
    ("a", "b", "g", "requires_move_around_goal", "expected"),
    [
        # Goal outside agent pair, no move around
        (1, 5, 0, False, 5),
        (1, 5, 6, False, 5),
        # Goal between agent pair, no move around
        (1, 5, 3, False, 2),
        (1, 5, 2, False, 3),
        # Goal outside agent pair, with move around
        (1, 5, 0, True, 5),
        (1, 5, 6, True, 5),
        # Goal between agent pair, with move around
        (1, 5, 3, True, 4),
        (1, 5, 2, True, 3),
        # TODO: add more edge cases: e.g. a, b, and g on the same pos
    ],
)
def test_calculate_time_in_one_axis(a, b, g, requires_move_around_goal, expected):
    assert calculate_time_in_one_axis(a, b, g, requires_move_around_goal) == expected


@pytest.mark.parametrize(
    ("agent1_location", "agent2_location", "goal_location", "expected_time"),
    [
        ((0, 3), (1, 2), (2, 1), 4),  # both agent in the same quadrant
        (
            (0, 3),
            (0, 0),
            (2, 1),
            4,
        ),  # adjacent qudarant (qudrant 2 and 3) => does not require moving around goal
        ((0, 2), (3, 0), (1, 1), 4),  # diagonal quadrant => requires moving around goal
    ],
)
def test_calcluate_optiaml_time_estimation_for_agent_pair(
    agent1_location: tuple[int, int],
    agent2_location: tuple[int, int],
    goal_location: tuple[int, int],
    expected_time: int,
):
    assert (
        calcluate_optiaml_time_estimation_for_agent_pair(
            agent1_location, agent2_location, goal_location
        )
        == expected_time
    )
