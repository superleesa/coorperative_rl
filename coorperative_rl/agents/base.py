from abc import ABC, abstractmethod
from typing import Any

from coorperative_rl.actions import Action
from coorperative_rl.states import AgentType, ObservableState


class BaseAgent(ABC):
    def __init__(self, agent_id: int, agent_type: AgentType) -> None:
        self.type = agent_type

        # used to ensure that the agent id is the same throughout the simulation
        # because self.state will be updated by the environment
        self.id = agent_id
        self.agent_type = agent_type

    @abstractmethod
    def decide_action(
        self,
        possible_actions: list[Action],
        state: ObservableState,
        is_training: bool = True,
    ) -> Action: ...

    @abstractmethod
    def update_model(
        self,
        original_state: ObservableState,
        moved_state: ObservableState,
        reward: float,
        action: Action,
    ) -> None: ...

    def update_hyper_parameters(
        self, current_episode_idx: int, num_total_episodes: int, **kwargs: Any
    ) -> None:
        pass

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseAgent):
            return False
        return hash(self) == hash(other)
