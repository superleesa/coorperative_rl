import pytest
from coorperative_rl.metrics import calculate_time_in_one_axis


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
