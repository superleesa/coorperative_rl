import numpy as np

from coorperative_rl.actions import Action
from coorperative_rl.states import ObservableState


class QTable:
    """
    Abstracts the Q-value matrix for the agent,
    to hide different q value matrices for different states and action handling
    """
    def __init__(self, n: int) -> None:
        # TODO: check item_location and goal_location are within the grid
        # TODO: the way we are stroing the q values is memory inefficient in a way that
        # not all state will have all actions (we are storing 0 for those)
        
        # these two matrices need to be different 
        # because q-value distribution is different before and after key share
        # i.e. before the agent has the key, it should take actions to meet the other agent
        # (agent_x, agent_y, closest_other_type_agent_x, closest_other_type_agent_y, goal_x, goal_y, action)
        self.until_key_share = np.zeros((n, n, n, n, n, n, len(Action)))
        self.after_key_share = np.zeros((n, n, n, n, len(Action)))  # opposite agent location should not matter after key share, so no extra (n, n) dimension
    
    def get_state_qvals(self, state: ObservableState, actions: list[Action] | Action = []) -> np.ndarray:
        """
        Returns Q(S), or Q(S, A) if actions are provided
        """
        if isinstance(actions, Action):
            actions = [actions]
        
        x, y = state.agent_location
        goal_x, goal_y = state.goal_location
        if state.has_full_key:
            return self.after_key_share[x, y, goal_x, goal_y] if not actions else self.after_key_share[x, y, goal_x, goal_y, [action.value for action in actions]]
        else:
            other_x, other_y = state.closest_opposite_agent_location
            return self.until_key_share[x, y, other_x, other_y, goal_x, goal_y] if not actions else self.until_key_share[x, y, other_x, other_y, goal_x, goal_y, [action.value for action in actions]]
    
    def update_qval(self, state: ObservableState, action: Action, new_qval: float) -> None:
        """
        Updates the Q value for a state-action pair, i.e. Q(S, A) = new_qval
        """
        x, y = state.agent_location
        goal_x, goal_y = state.goal_location
        if state.has_full_key:
            self.after_key_share[x, y, goal_x, goal_y, action.value] = new_qval
        else:
            other_x, other_y = state.closest_opposite_agent_location
            self.until_key_share[x, y, other_x, other_y, goal_x, goal_y, action.value] = new_qval
    
    def increase_qval(self, state: ObservableState, action: Action, increment: float) -> None:
        """
        Increases the Q value for a state-action pair, i.e. Q(S, A) += increment
        """
        x, y = state.agent_location
        goal_x, goal_y = state.goal_location
        if state.has_full_key:
            self.after_key_share[x, y, goal_x, goal_y, action.value] += increment
        else:
            other_x, other_y = state.closest_opposite_agent_location
            self.until_key_share[x, y, other_x, other_y, goal_x, goal_y, action.value] += increment