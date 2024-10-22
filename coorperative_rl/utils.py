from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coorperative_rl.agents.base import BaseAgent
    from coorperative_rl.states import AgentType

def generate_random_location(grid_size: int, disallowed_locations: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    return generate_random_pair_numbers(0, grid_size-1, disallowed_locations)


def generate_random_pair_numbers(min_val: int, max_val: int, disallowed_pairs: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    if disallowed_pairs is None:
        disallowed_pairs = []
    
    while True:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        pair = (a, b)
        
        if pair in disallowed_pairs:
            continue
        
        return pair


def generate_grid_location_list(max_x: int, max_y) -> list[tuple[int, int]]:
    """
    Generate the grid location list for all possible cases
    """
    return [(i, j) for i in range(max_x) for j in range(max_y)]


def shuffle_list_not_in_place(lst: list) -> list:
    """
    Shuffle the list without changing the original list
    """
    return random.sample(lst, len(lst))

def flatten_2D_list(lst: list) -> list:
    """
    Flatten 2D list to 1D list
    """
    return [item for sublist in lst for item in sublist]

def shuffle_and_distribute_agents(agents: list[BaseAgent]) -> list[BaseAgent]:
    """
    Returns a list, where there is 1 agent from each type in the begginng, and then remaining agents in random order
    
    [type_a_agent, type_b_agent, ......]
    1 agent from each type, then any order
    """
    
    type_to_agents: dict[AgentType, list[BaseAgent]] = {}
    for agent in agents:
        if agent.type not in type_to_agents:
            type_to_agents[agent.type] = []
        type_to_agents[agent.type].append(agent)
    
    # FIXME: we need to ensure there is order in agent types, or else it's not "fixed"
    fixed_agents: list[BaseAgent] = []
    for agents in type_to_agents.values():
        random_index = random.randint(0, len(agents)-1)
        fixed_agents.append(agents.pop(random_index))  # maybe there is a better way to do this
    
    return fixed_agents + shuffle_list_not_in_place(flatten_2D_list(list(type_to_agents.values())))
