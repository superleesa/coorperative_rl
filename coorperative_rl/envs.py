from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from coorperative_rl.states import GlobalState, AgentState, ObservableState

from coorperative_rl.actions import Action
from coorperative_rl.map_visualizer import MapVisualizer

if TYPE_CHECKING:
    from coorperative_rl.agents.base import BaseAgent


class Environment:
    def __init__(
        self,
        grid_size: int = 5,
        goal_state_reward: int | float = 50,
        key_share_reward: int | float = 100,
        goal_without_key_penalty: int | float = -100,
        time_penalty: int | float = -1,
        visualize: bool = True,
    ) -> None:
        """
        
        Args:
            grid_size (int, optional): The size of the square grid. Defaults to 5.
            goal_state_reward (int | float, optional): The reward for reaching the goal state with the key. Defaults to 50.
            key_share_reward (int | float, optional): The reward for sharing the key. Defaults to 100.
            goal_without_key_penalty (int | float, optional): The penalty for reaching the goal without the key. Defaults to -100.
            time_penalty (int | float, optional): The penalty for each time step. Defaults to -1.
            visualize (bool, optional): Whether to visualize the agents interactions and the environment. Defaults to True.
        """
        self.grid_size = grid_size
        self.state = GlobalState(grid_size=grid_size)
        self.visualizer = MapVisualizer(self, visualize=visualize)

        self.time_penalty = time_penalty
        self.goal_state_reward = goal_state_reward
        self.goal_without_key_penalty = goal_without_key_penalty
        self.key_share_reward = key_share_reward

        # used to calculate the reward across agent actions
        self.prev_step_state = deepcopy(self.state)
        self.action_taken = {}

    def add_agent(self, agent: BaseAgent) -> None:
        """
        Adds an agent to the environment. 
        Before running episodes using the environment, you must subscribe agents to the environment.
        
        Args:
            agent (BaseAgent): The agent to add to the environment.
        """
        self.state.add_agent(agent)

    def remove_agent(self, agent: BaseAgent) -> None:
        """
        Remove an agent from the environment.
        If the agent should no longer be part of the environment, you can remove it.
        
        Args:
            agent (BaseAgent): The agent to remove from the environment.
        """
        self.state.agents.remove(agent)
        if agent in self.state.agent_states:
            del self.state.agent_states[agent]

    def initialize_for_new_episode(
        self,
        agent_states: dict[BaseAgent, AgentState] | None = None,
        goal_location: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the environment for a new episode.
        You can either initialize the environment with specific states, or randomly.
        
        Args:
            agent_states (dict[BaseAgent, AgentState], optional): A dictionary of agent states. Defaults to None.
            goal_location (tuple[int, int], optional): The location of the goal state. Defaults to None.
            **kwargs: Additional keyword arguments passed to the state initialization function.
        """
        if (
            agent_states is None
            and goal_location is not None
            or agent_states is not None
            and goal_location is None
        ):
            raise ValueError("not supported (yet)")

        if agent_states is not None and goal_location is not None:
            self.state.initialize_state_from_values(
                agent_states, goal_location, **kwargs
            )
        else:
            self.state.initialize_state_randomly(**kwargs)

    def get_available_actions(self, agent: BaseAgent) -> list[Action]:
        """
        A function to rule out the actions that are not available to the agent,
        based on obserbable state and assumptions. 
        For example, if the agent is at the edge of the grid, it cannot move in that direction.
        
        Args:
            agent (BaseAgent): The agent for which to get the available actions.
        
        Returns:
            list[Action]: A list of actions that the agent can take.
        """

        observable_state = self.state.get_observable_state_for_agent(agent)

        actions = []

        # assume agents know waiting after the key share is meaningless
        if not observable_state.has_full_key:
            actions.append(Action.WAIT)

        x, y = observable_state.agent_location
        # we assume that agents know the grid size and boundaries
        if x > 0:
            actions.append(Action.LEFT)  # left
        if x < self.grid_size - 1:
            actions.append(Action.RIGHT)  # right
        if y > 0:
            actions.append(Action.DOWN)  # down
        if y < self.grid_size - 1:
            actions.append(Action.UP)  # up

        return actions

    def apply_action_to_state(self, agent_state: AgentState, action: Action) -> None:
        """
        Mutates the agent state by applying the action.
        It does not necessary only modify the given agent state but also the other agent states.
        
        Args:
            agent_state (AgentState): The agent state to apply the action to.
            action (Action): The action to apply to the agent.
        """
        # FIXME
        x, y = agent_state.location

        # Check each action and ensure it stays within the bounds
        if action == Action.LEFT:
            if x > 0:  # Ensure not moving out of bounds on the left
                agent_state.location = (x - 1, y)
        elif action == Action.RIGHT:
            if x < self.grid_size - 1:  # Ensure not moving out of bounds on the right
                agent_state.location = (x + 1, y)
        elif action == Action.DOWN:
            if y > 0:  # Ensure not moving out of bounds downwards
                agent_state.location = (x, y - 1)
        elif action == Action.UP:
            if y < self.grid_size - 1:  # Ensure not moving out of bounds upwards
                agent_state.location = (x, y + 1)
        elif action == Action.WAIT:
            # do not move if wait
            pass
        else:
            raise ValueError("Invalid action")

        # if there is an opposite agent at the same location (and they both do not hold a full key yet),
        # both agents share the key => has full key
        if (
            not agent_state.has_full_key
            and (other_agent_state := self.get_opposite_agent_at(agent_state))
            is not None
            and not other_agent_state.has_full_key
        ):
            agent_state.has_full_key = True
            other_agent_state.has_full_key = True

    def get_opposite_agent_at(self, agent_state: AgentState) -> AgentState | None:
        """
        Returns the first opposite agent found
        (so agent added to the environment first will have priority)
        FIXME: this cause bias because it's not something agent should learn
        
        Args:
            agent_state (AgentState): The agent state to check for opposite agents.
        
        Returns:
            AgentState | None: The opposite agent state if found, otherwise None.
        """

        for _, other_agent_state in self.state.agent_states.items():
            if (
                other_agent_state.type != agent_state.type
                and other_agent_state.location == agent_state.location
            ):
                return other_agent_state

        return None

    def get_reward(self) -> dict[BaseAgent, float]:
        """
        Implements R(s, a, s') function.
        Returns the reward for each agent based on the state transition.
        
        Returns:
            dict[BaseAgent, float]: A dictionary of agent
        """
        # TODO: maybe add reencounter penalty

        agent_to_sas: dict[BaseAgent, tuple[AgentState, Action, AgentState]] = {}
        for agent in self.state.agents:
            prev_state = self.prev_step_state.agent_states[agent]
            next_state = self.state.agent_states[agent]
            agent_to_sas[agent] = (prev_state, self.action_taken[agent], next_state)

        agent_to_reward: dict[BaseAgent, float] = {
            agent: self.time_penalty for agent in agent_to_sas.keys()
        }

        # add high reward to contributed agents only if they share the key
        for agent, (prev_state, _, current_state) in agent_to_sas.items():
            if not prev_state.has_full_key and current_state.has_full_key:
                agent_to_reward[agent] += (
                    self.key_share_reward
                )  # each contributor gets the same reward

        # add high reward to the agent that reaches the goal with the key
        for agent, (prev_state, _, current_state) in agent_to_sas.items():
            if (
                current_state.location == self.state.goal_location
                and current_state.has_full_key
            ):
                agent_to_reward[agent] += self.goal_state_reward

        # penalize agents for reaching the goal without the key
        for agent, (prev_state, _, current_state) in agent_to_sas.items():
            if (
                current_state.location == self.state.goal_location
                and not current_state.has_full_key
            ):
                agent_to_reward[agent] += self.goal_without_key_penalty

        return agent_to_reward

    def is_done(self) -> bool:
        """
        Returns whether the episode is done.
        If at least one agent has reached the goal with the key, the episode is done.
        
        Returns:
            bool: True if the episode is done, otherwise False.
        """
        return any(
            [
                agent_state.location == self.state.goal_location
                and agent_state.has_full_key
                for agent_state in self.state.agent_states.values()
            ]
        )

    def start_new_step(self) -> None:
        """
        Call this function at the beginning of each step.
        """
        self.prev_step_state = deepcopy(self.state)
        self.action_taken = {}

    def step_one_agent(
        self, agent: BaseAgent, action: Action
    ) -> tuple[ObservableState, float]:
        """
        Interact one agent with the environment by taking an action.
        Note that this function does not return rewards
        because rewards must be computed after all agents take their actions.
        """
        self.apply_action_to_state(self.state.agent_states[agent], action)
        self.action_taken[agent] = action
        self.visualizer.update()

        # if one agent reaches the goal state, don't update anymore,
        # though remaining agents in the current step will still take their actions
        # (because users are not interested in the rest of the episode)
        if not self.is_done():
            self.visualizer.update()
        return self.state.get_observable_state_for_agent(agent)

    def end_step(
        self,
    ) -> tuple[
        dict[BaseAgent, ObservableState],
        dict[BaseAgent, ObservableState],
        dict[BaseAgent, float],
        bool,
    ]:
        """
        Call this function at the end of each step.
        It calculates reward and collectes SARS' and whether the episode is done.
        
        Returns:
            tuple[dict[BaseAgent, ObservableState], dict[BaseAgent, ObservableState], dict[BaseAgent, float], bool]: The observable states of agents before and after the step, the rewards for each agent, and whether the episode is done.
        """
        rewards = self.get_reward()
        is_done = self.is_done()
        return (
            {
                agent: self.prev_step_state.get_observable_state_for_agent(agent)
                for agent in self.state.agents
            },
            {
                agent: self.state.get_observable_state_for_agent(agent)
                for agent in self.state.agents
            },
            rewards,
            is_done,
        )
