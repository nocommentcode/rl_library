import argparse
from typing import List
from agents.FeatureEncoderParams import FeatureEncoderParams
import torch.nn as nn
import torch


class DeepQParams(FeatureEncoderParams):
    target_update_steps: int
    agent_name = "DQN"
    fc: List[int]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('-tu', '--target_update_steps', type=int,
                            default=10,
                            help='Sync target network every n steps')
        parser.add_argument('-fc', '--fully_connected', type=int, nargs='+',
                            default=[32],
                            help='Fully connected layer output features')

    def set_args(self, **kwargs) -> None:
        super().set_args(**kwargs)
        self.target_update_steps = kwargs['target_update_steps']
        self.fc = kwargs['fully_connected'] if 'fully_connected' in kwargs \
            else kwargs['fc']

    def build_model(self) -> nn.Module:
        """
        Builds the model based on the parameters
        """
        env = self.make_env()
        observation_space = env.observation_space
        state_space = observation_space.shape
        n_actions = env.action_space.n

        model = self.build_encoder()
        ouput_size = model(torch.zeros(1, *state_space)).shape[1]

        # add fully connected layers
        in_features = ouput_size
        for out_features in self.fc[1:]:
            model.append(nn.Linear(in_features=in_features,
                                   out_features=out_features))
            model.append(nn.ReLU())
            in_features = out_features

        # add output layer
        model.append(nn.Linear(in_features=in_features,
                               out_features=n_actions))
        model.to(self.device)
        return model
