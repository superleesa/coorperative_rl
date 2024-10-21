from __future__ import annotations

import random
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from coorperative_rl.utils import generate_random_location

if TYPE_CHECKING:
    from coorperative_rl.agents.base import BaseAgent


class AgentType(Enum):
    TYPE_A = 0
    TYPE_B = 1


class AgentState(BaseModel):
    id: int = Field(frozen=True)
    type: AgentType = Field(frozen=True)
    location: tuple[int, int]
    has_full_key: bool


class ObservableState(BaseModel):
    """
    This is the state that agents can observe and handle, 
    when decicing thier actions / updating thier models
    """
    model_config = ConfigDict(frozen=True)  # disable mutation for simplicity
    
    # we can only use:
    # 1) the agent location itself
    # 2) agent location of the closest opposite type agent (this one from the contract)
    # 3) whether the agent itself has full key or not

    agent_location: tuple[int, int]
    closest_opposite_agent_location: tuple[int, int]
    has_full_key: bool
    goal_location: tuple[int, int]


class GlobalState:
    def __init__(
        self,
        grid_size: int,
    ) -> None:
        self.grid_size = grid_size
        self.agents: set[BaseAgent] = set([])
        self.agent_states: dict[BaseAgent, AgentState] = {}
        self.goal_location: tuple[int, int]

    def get_closest_opposite_agent(self, agent_state: AgentState) -> AgentState:
        min_distance = float("inf")
        min_distance_agent_id: int | None = None

        for other_agent_id, other_agent_state in self.agent_states.items():
            if other_agent_state.type == agent_state.type:
                continue

            euclidan_distance = (
                (other_agent_state.location[0] - agent_state.location[0]) ** 2
                + (other_agent_state.location[1] - agent_state.location[1]) ** 2
            ) ** (1 / 2)
            if euclidan_distance < min_distance:
                min_distance = euclidan_distance
                min_distance_agent_id = other_agent_id

        if min_distance_agent_id is None:
            raise ValueError("No opposite agent found")

        return self.agent_states[min_distance_agent_id]

    def get_observable_state_for_agent(self, agent: BaseAgent) -> ObservableState:
        agent_state = self.agent_states[agent]
        closest_opposite_agent_state = self.get_closest_opposite_agent(agent_state)
        return ObservableState(
            agent_location=agent_state.location,
            closest_opposite_agent_location=closest_opposite_agent_state.location,
            has_full_key=agent_state.has_full_key,
            goal_location=self.goal_location,
        )

    def add_agent(self, agent: BaseAgent) -> None:
        if agent in self.agents:
            raise ValueError("Agent with the same id already exists")
        self.agents.add(agent)

    def initialize_state_randomly(
        self, allow_overlapping_objects: bool = False, has_full_key_prob: float = 0.0
    ) -> None:
        """
        this overwrites the agent states (so don't reference the same agent state across episodes)

        Args:
            allow_overlapping_objects: whether to allow agents / goal to be placed at the same location. 
                Defaults to False. during training it might be helpful to set this to true for fairer sampling
            has_full_key_prob: probability of having the full key. Defaults to 0. 
                During training it might be helpful to raise this probability to specifically train the paths of agents after key sharing
        """
        selected_locations = []

        num_locations_needed = len(self.agents) + 1  # agents + goal
        for _ in range(num_locations_needed):
            location = generate_random_location(
                self.grid_size,
                selected_locations if not allow_overlapping_objects else None,
            )
            selected_locations.append(location)

        self.goal_location = selected_locations.pop()

        for agent in self.agents:
            agent_state = AgentState(
                id=agent.id,
                type=agent.type,
                location=selected_locations.pop(),
                has_full_key=random.random() <= has_full_key_prob
                if has_full_key_prob is not None
                else False,
            )
            self.agent_states[agent] = agent_state

    def initialize_state_from_values(
        self,
        agent_states: dict[BaseAgent, AgentState],
        goal_location: tuple[int, int],
    ) -> None:
        self.goal_location = goal_location
        self.agent_states = agent_states
        self.agents = set(agent_states.keys())
