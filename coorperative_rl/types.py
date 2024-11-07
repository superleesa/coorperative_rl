from typing import TypedDict, TypeAlias

from coorperative_rl.agents.base import BaseAgent
from coorperative_rl.states import AgentState, ObservableState
from coorperative_rl.actions import Action


# TODO: port these types somewhere else
SARS: TypeAlias = tuple[
    dict[BaseAgent, ObservableState],
    dict[BaseAgent, Action],
    dict[BaseAgent, float],
    dict[BaseAgent, ObservableState],
]


class EpisodeSampleParams(TypedDict):
    agent_states: dict[BaseAgent, AgentState]
    goal_location: tuple[int, int]


def serialize_sars(sars: SARS) -> dict:
    """
    Serialize the SARS tuple to EpisodeSampleParams
    """
    s, a, r, s_ = sars
    s_serialized = {agent.id: state.model_dump() for agent, state in s.items()}
    a_serialized = {agent.id: action.value for agent, action in a.items()}
    r_serialized = {agent.id: reward for agent, reward in r.items()}
    s_serialized_ = {agent.id: state.model_dump() for agent, state in s_.items()}
    
    return {
        "s": s_serialized,
        "a": a_serialized,
        "r": r_serialized,
        "s_": s_serialized_,
    }


def deserialize_sars(sars_serialized: dict) -> SARS:
    """
    Deserialize the SARS
    
    Args:
        sars_serialized: the serialized SARS tuple
        agents: the agents dict to map agent ids to agents
    """
    s = {agent_id: ObservableState(**state) for agent_id, state in sars_serialized["s"].items()}
    a = {agent_id: Action(action_id) for agent_id, action_id in sars_serialized["a"].items()}
    r = {agent_id: reward for agent_id, reward in sars_serialized["r"].items()}
    s_ = {agent_id: ObservableState(**state) for agent_id, state in sars_serialized["s_"].items()}
    
    return s, a, r, s_
