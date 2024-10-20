from abc import ABC, abstractmethod

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
    ) -> None:
        ...

    def __hash__(self) -> int:
        return self.id
