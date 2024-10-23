from abc import ABC, abstractmethod
from typing import Any

from coorperative_rl.actions import Action
from coorperative_rl.states import AgentType, ObservableState


class BaseAgent(ABC):
    """
    Base class for all agents in the environment. 
    Agents should have the ability to decide actions, update their models, and update hyperparameters.
    """
    def __init__(self, agent_id: int, agent_type: AgentType) -> None:
        """
        Initializes a new instance of the agent.
        Args:
            agent_id: The unique identifier for the agent.
            agent_type: The type of the agent.
        """
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
    ) -> Action:
        """
        Decides the action to take from a list of possible actions based on the current state.

        Args:
            possible_actions: A list of possible actions to choose from.
            state: The current observable state of the environment.
            is_training: Flag indicating whether the agent is in training mode. Defaults to True.

        Returns:
            The action decided by the agent.
        """
        ...

    @abstractmethod
    def update_model(
        self,
        original_state: ObservableState,
        moved_state: ObservableState,
        reward: float,
        action: Action,
    ) -> None: 
        """
        Updates the model based on the transition from the original state to the moved state.
        You don't necessarily have to alway update the model e.g. if you are implementing DQN, 
        you can just store the transition in the replay buffer in some cases.

        Args:
            original_state: The state before the action was taken.
            moved_state: The state after the action was taken.
            reward: The reward received after taking the action.
            action: The action that was taken.
        """
        ...
    

    def update_hyper_parameters(
        self, current_episode_idx: int, num_total_episodes: int, **kwargs: Any
    ) -> None:
        """
        Update the hyperparameters of the agent, based on the training state.

        Args:
            current_episode_idx: The index of the current episode.
            num_total_episodes: The total number of episodes.
            **kwargs: Additional keyword arguments for hyperparameter updates.
        """
        pass

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        """
        We implement this method because we use BaseAgent as a key in dictionaries.
        """
        if not isinstance(other, BaseAgent):
            return False
        return hash(self) == hash(other)
