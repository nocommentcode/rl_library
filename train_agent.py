import argparse
from agent_factory import detect_selected_agent, get_agent, get_agent_params


def train_agent(agent_name: str):
    parser = argparse.ArgumentParser()

    parser.add_argument('-si', '--save_every_n', type=int,
                        default=300,
                        help='Save model every n episodes (-1 to disable)')

    agent_params = get_agent_params(agent_name)
    agent_params.add_arguments(parser)

    args, _ = parser.parse_known_args()
    agent_params.parse_args(args)

    print(str(args))
    print(f"Start trianing on {args.device}")

    def save_callback(episode: int):
        if args.save_every_n > 0 and episode % args.save_every_n == 0:
            agent.save(f"{agent_params.save_dir}/checkpoints/{episode}")

    agent = get_agent(agent_name)(agent_params)
    agent.to(args.device)
    agent.train(callback=save_callback)

    print("Finished training")
    agent.save(agent_params.save_dir)


if __name__ == "__main__":
    agent_name = detect_selected_agent()
    train_agent(agent_name)
