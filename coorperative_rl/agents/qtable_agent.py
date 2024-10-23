import random
from typing import Any

import numpy as np

from coorperative_rl.actions import Action
from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import AgentType, ObservableState
from coorperative_rl.agents.qtable import QTable


class QTableAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        qval_matrix: QTable,
        epsilon_initial: float | None = 0.9,
        epsilon_final: float | None = 0.1,
        alpha: float | None = 0.3,
        discount_rate: float | None = 0.9,
    ) -> None:
        super().__init__(agent_id, agent_type)

        self.qval_matrix = qval_matrix

        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon = epsilon_initial
        self.alpha = alpha
        self.discount_rate = discount_rate

    def decide_action(
        self,
        possible_actions: list[Action],
        state: ObservableState,
        is_training: bool = True,
    ) -> Action:
        """
        Epislon greedy method to choose action
        """
        if is_training and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            action_to_qval = list(
                zip(
                    possible_actions,
                    self.qval_matrix.get_state_qvals(state, actions=possible_actions),
                )
            )
            random.shuffle(action_to_qval)  # to break ties randomly
            return max(action_to_qval, key=lambda x: x[1])[0]

    def update_model(
        self,
        original_state: ObservableState,
        moved_state: ObservableState,
        reward: float,
        action: Action,
    ) -> None:
        if self.discount_rate is None:
            raise ValueError(
                "discount_rate must be provided when initialization of agent to update model"
            )

        qval_difference: float = self.alpha * (
            reward
            + self.discount_rate * np.max(self.qval_matrix.get_state_qvals(moved_state))
            - self.qval_matrix.get_state_qvals(original_state, actions=action)
        )
        self.qval_matrix.increase_qval(original_state, action, qval_difference)

    def update_hyper_parameters(
        self, current_episode_idx: int, num_total_episodes: int, **kwargs: Any
    ) -> None:
        if (
            self.epsilon_initial is None
            or self.epsilon_final is None
            or self.alpha is None
        ):
            raise ValueError(
                "epsilon_initial, epsilon_final, and alpha must be provided to update hyper parameters"
            )

        # linear epsilon decay
        epsilon = self.epsilon_initial - (
            (self.epsilon_initial - self.epsilon_final)
            * (current_episode_idx / num_total_episodes)
        )
        self.epsilon = max(
            self.epsilon_final, epsilon
        )  # Ensure epsilon doesn't go below epsilon_final
