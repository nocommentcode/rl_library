import argparse
from agents.RLAgent import RLAgent
from agents.a2c.A2CParams import A2CParams
from agents.ddpg.DDPGParams import DDPGParams
from agents.ddpg.DDPGAgent import DDPGAgent
from agents.dqn.DeepQParams import DeepQParams
from agents.dqn.DeepQAgent import DeepQAgent
from agents.a2c.A2CAgent import A2CAgent

AGENTS = ['DQN', 'A2C', 'DDPG']

PARAMS = {
    'DQN': DeepQParams,
    'A2C': A2CParams,
    'DDPG': DDPGParams,
}

AGENT_CLASSES = {
    'DQN': DeepQAgent,
    'A2C': A2CAgent,
    'DDPG': DDPGAgent,
}


def get_agent_params(agent_name: str):
    """
    Returns the agent parameters for the given agent name

    Args:
        agent_name: the name of the agent

    Returns:
        the agent parameters
    """
    if agent_name in PARAMS:
        return PARAMS[agent_name]()
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
    if agent_name in AGENT_CLASSES:
        return AGENT_CLASSES[agent_name]
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


def load_agent(agent_name: str, dir_path: str) -> RLAgent:
    """
    Loads the agent from the given directory

    Args:
        name: the name of the agent
        dir_path: the directory to load from

    Returns:
        the agent
    """
    agent = get_agent(agent_name).load(dir_path)
    agent.to(agent.args.device)
    return agent
