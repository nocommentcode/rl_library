import argparse
from agents.DiscreteActionAgentParams import DiscreteActionAgentParams


class DeepQParams(DiscreteActionAgentParams):
    target_update_steps: int
    agent_name = "DQN"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('-tu', '--target_update_steps', type=int,
                            default=10,
                            help='Sync target network every n steps')

    def set_args(self, **kwargs) -> None:
        super().set_args(**kwargs)
        self.sync_target_every_n_episodes = kwargs['target_update_steps']
