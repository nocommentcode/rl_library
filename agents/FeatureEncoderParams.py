import argparse
from typing import List

from agents.RLAgentParams import RLAgentParams
import torch.nn as nn
import torch


class FeatureEncoderParams(RLAgentParams):
    """
    Base class for all agents that take discrete actions
    Adds arguments for the model parameters and builds the model
    """

    convolutions: List[int]
    kernel_size: List[int]
    stride: List[int]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('-c', '--convolutions', type=int, nargs='+',
                            default=[32],
                            help='Convolution layer output channels')
        parser.add_argument('-k', '--kernel_size', type=int, nargs='+',
                            default=[3], help='Convolution kernel size')
        parser.add_argument('-st', '--stride', type=int, nargs='+',
                            default=[1], help='Convolution stride')

    def set_args(self, **kwargs) -> None:
        super().set_args(**kwargs)
        self.convolutions = kwargs['convolutions']
        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride']

    def build_encoder(self) -> nn.Module:
        """
        Builds the feature encoder based on the parameters
        """

        encoder = nn.Sequential()

        env = self.make_env()
        observation_space = env.observation_space
        state_space = observation_space.shape

        in_channels = state_space[0]
        for out_channel, stride, kernal in zip(self.convolutions,
                                               self.stride,
                                               self.kernel_size):
            encoder.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channel,
                          kernel_size=kernal,
                          stride=stride))
            encoder.append(nn.ReLU())
            in_channels = out_channel

        encoder.append(nn.Flatten())
        encoder.to(self.device)
        return encoder
