import argparse
from agents.RLAgent import RLAgent
from agents.a2c.A2CParams import A2CParams
from agents.dqn.DeepQParams import DeepQParams


AGENTS = ['DQN', 'A2C']


def get_agent_params(agent_name: str):
    """
    Returns the agent parameters for the given agent name

    Args:
        agent_name: the name of the agent

    Returns:
        the agent parameters
    """
    if agent_name == 'DQN':
        return DeepQParams()
    elif agent_name == 'A2C':
        return A2CParams()
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")


def get_agent(agent_name: str) -> RLAgent:
    """
    Returns the agent for the given agent name

    Args:
        agent_name: the name of the agent

    Returns:
        the agent
    """
    if agent_name == 'DQN':
        from agents.dqn.DeepQAgent import DeepQAgent
        return DeepQAgent
    elif agent_name == 'A2C':
        from agents.a2c.A2CAgent import A2CAgent
        return A2CAgent
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")


def detect_selected_agent() -> str:
    """
    Detects the selected agent from the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str,
                        required=True, choices=AGENTS)
    args, _ = parser.parse_known_args()
    return args.agent_type
