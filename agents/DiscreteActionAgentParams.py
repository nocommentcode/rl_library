import argparse
from typing import List

from agents.RLAgentParams import RLAgentParams
import torch.nn as nn
import torch


class DiscreteActionAgentParams(RLAgentParams):
    """
    Base class for all agents that take discrete actions
    Adds arguments for the model parameters and builds the model
    """

    convolutions: List[int]
    kernel_size: List[int]
    stride: List[int]
    fc: List[int]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('-c', '--convolutions', type=int, nargs='+',
                            default=[32],
                            help='Convolution layer output channels')
        parser.add_argument('-k', '--kernel_size', type=int, nargs='+',
                            default=[3], help='Convolution kernel size')
        parser.add_argument('-st', '--stride', type=int, nargs='+',
                            default=[1], help='Convolution stride')
        parser.add_argument('-fc', '--fully_connected', type=int, nargs='+',
                            default=[32],
                            help='Fully connected layer output features')

    def set_args(self, **kwargs) -> None:
        super().set_args(**kwargs)
        self.convolutions = kwargs['convolutions']
        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride']
        self.fc = kwargs['fully_connected'] if 'fully_connected' in kwargs \
            else kwargs['fc']

    def build_model(self) -> nn.Module:
        """
        Builds the model based on the parameters
        """

        model = nn.Sequential()

        env = self.make_env()
        observation_space = env.observation_space
        state_space = observation_space.shape
        n_actions = env.action_space.n

        # conv layers
        in_channels = state_space[0]
        for out_channel, stride, kernal in zip(self.convolutions,
                                               self.stride,
                                               self.kernel_size):
            model.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channel,
                          kernel_size=kernal,
                          stride=stride))
            model.append(nn.ReLU())
            # model.append(nn.MaxPool2d(kernel_size=kernal))
            in_channels = out_channel

        model.append(nn.Flatten())

        # fc layers
        in_features = model(torch.zeros(1, *state_space)).shape[1]
        for out_features in self.fc:
            model.append(torch.nn.Linear(
                in_features, out_features, bias=True))
            model.append(torch.nn.ReLU())
            in_features = out_features

        # final layer
        model.append(torch.nn.Linear(in_features, n_actions))
        model.to(self.device)
        return model
