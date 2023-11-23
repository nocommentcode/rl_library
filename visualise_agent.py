import argparse
import time

from agent_factory import detect_selected_agent, get_agent


def visualize_episode(agent_name: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--run_dir',
                        type=str, required=True,
                        help='Run directory to load agent from')
    parser.add_argument('-ml', '--max_length', type=int, default=1000,
                        help='Maximum number of steps to run for')
    args = parser.parse_args()

    agent = get_agent(agent_name).load(args.run_dir)
    agent.to(agent.args.device)

    agent.visualize(seed=int(time.time()), max_length=args.max_length)


if __name__ == "__main__":
    agent_name = detect_selected_agent()
