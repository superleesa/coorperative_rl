from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
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
        self.state.add_agent(agent)

    def initialize_for_new_episode(
        self,
        agent_states: dict[BaseAgent, AgentState] | None = None,
        goal_location: tuple[int, int] | None = None,
        allow_overlapping_objects: bool = False,
    ) -> None:
        if (
            agent_states is None
            and goal_location is not None
            or agent_states is not None
            and goal_location is None
        ):
            raise ValueError("not supported (yet)")

        if agent_states is not None and goal_location is not None:
            if allow_overlapping_objects:
                raise ValueError("overlapping objects are not supported with custom agent states")
            self.state.initialize_state_from_values(agent_states, goal_location)
        else:
            self.state.initialize_state_randomly(allow_overlapping_objects=allow_overlapping_objects)

    def get_available_actions(self, agent: BaseAgent) -> list[Action]:
        """
        A function to rule out the actions that are not available to the agent,
        based on obserbable state and assumptions
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
        Mutates the agent state by applying the action
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
        returns the first opposite agent found
        (so agent added to the environment first will have priority)
        FIXME: this cause bias because it's not something agent should learn
        """

        for _, other_agent_state in self.state.agent_states.items():
            if (
                other_agent_state.type != agent_state.type
                and other_agent_state.location == agent_state.location
            ):
                return other_agent_state

        return None

    def get_reward(self) -> dict[BaseAgent, float]:
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
        If at least one agent has reached the goal with the key, the episode is done
        """
        return any([agent_state.location == self.state.goal_location and agent_state.has_full_key for agent_state in self.state.agent_states.values()])

    def start_new_step(self) -> None:
        self.prev_step_state = deepcopy(self.state)
        self.action_taken = {}

    def step_one_agent(
        self, agent: BaseAgent, action: Action
    ) -> tuple[ObservableState, float]:
        self.apply_action_to_state(self.state.agent_states[agent], action)
        self.action_taken[agent] = action
        self.visualizer.update()
        return self.state.get_observable_state_for_agent(agent)

    def end_step(self) -> tuple[dict[BaseAgent, ObservableState], dict[BaseAgent, ObservableState], dict[BaseAgent, float], bool]:
        """
        
        Returns: previous state, current state, rewards, is_done
        """
        rewards = self.get_reward()
        is_done = self.is_done()
        return (
            {agent: self.prev_step_state.get_observable_state_for_agent(agent) for agent in self.state.agents},
            {
                agent: self.state.get_observable_state_for_agent(agent)
                for agent in self.state.agents
            },
            rewards,
            is_done,
        )
