import argparse
from agent_factory import detect_selected_agent, get_agent, get_agent_params, load_agent
from agents.RLAgent import RLAgent


def make_save_callback(save_freq: int, agent: RLAgent):
    def save_callback(episode: int):
        if save_freq > 0 and episode % save_freq == 0:
            agent.save(f"{agent.save_dir}/checkpoints/{episode}")
    return save_callback


def train_agent(agent_name: str):
    parser = argparse.ArgumentParser()

    parser.add_argument('-si', '--save_every_n', type=int,
                        default=300,
                        help='Save model every n episodes (-1 to disable)')
    parser.add_argument('-ctd', '--continue_training_dir', type=str,
                        default=None,
                        help='Continue training from the given directory')

    args, _ = parser.parse_known_args()

    if args.continue_training_dir is not None:
        agent = load_agent(agent_name, args.continue_training_dir)
        agent_params = agent.args

        print(str(agent_params))
        print(f"Coninue trianing on {args.device}")
    else:

        agent_params = get_agent_params(agent_name)
        agent_params.add_arguments(parser)

        args, _ = parser.parse_known_args()
        agent_params.parse_args(args)

        agent = get_agent(agent_name)(agent_params)
        agent.to(agent_params.device)

        print(str(agent_params))
        print(f"Start trianing on {agent_params.device}")

    agent.train(callback=make_save_callback(args.save_every_n, agent))

    print("Finished training")
    agent.save(agent_params.save_dir)


if __name__ == "__main__":
    agent_name = detect_selected_agent()
    train_agent(agent_name)
