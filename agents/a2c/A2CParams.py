from agents.DiscreteActionAgentParams import DiscreteActionAgentParams

import torch
import torch.nn as nn

import argparse
from typing import List, Tuple


class A2CParams(DiscreteActionAgentParams):
    actor_fc: List[int]
    critic_fc: List[int]
    agent_name = "A2C"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument('-afc', '--actor_fully_connected', type=int,
                            nargs='+',
                            default=[32],
                            help='Actor fully connected layer output features')
        parser.add_argument('-cfc', '--critic_fully_connected', type=int,
                            nargs='+',
                            default=[32],
                            help='Critic fully connected layer output features')

    def set_args(self, **kwargs) -> None:
        super().set_args(**kwargs)
        self.actor_fc = kwargs['actor_fully_connected'] if 'actor_fully_connected' in kwargs \
            else kwargs['afc']
        self.critic_fc = kwargs['critic_fully_connected'] if 'critic_fully_connected' in kwargs \
            else kwargs['cfc']

    def build_model(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Builds the model based on the parameters

        Returns:
            shared_conv: The shared convolutional layers
            actor: The actor network
            critic: The critic network
        """

        shared_conv = nn.Sequential()

        env = self.make_env()
        observation_space = env.observation_space
        state_space = observation_space.shape
        n_actions = env.action_space.n

        # conv layers
        in_channels = state_space[0]
        for out_channel, stride, kernal in zip(self.convolutions,
                                               self.stride,
                                               self.kernel_size):
            shared_conv.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channel,
                          kernel_size=kernal,
                          stride=stride))
            shared_conv.append(nn.ReLU())
            in_channels = out_channel

        shared_conv.append(nn.Flatten())

        conv_output_features = shared_conv(
            torch.zeros(1, *state_space)).shape[1]

        # critic
        critic = nn.Sequential()
        in_features = conv_output_features
        for out_features in self.critic_fc:
            critic.append(torch.nn.Linear(
                in_features, out_features, bias=True))
            critic.append(torch.nn.ReLU())
            in_features = out_features
        critic.append(torch.nn.Linear(in_features, 1))

        # actor
        actor = nn.Sequential()
        in_features = conv_output_features
        for out_features in self.actor_fc:
            actor.append(torch.nn.Linear(
                in_features, out_features, bias=True))
            actor.append(torch.nn.ReLU())
            in_features = out_features
        actor.append(torch.nn.Linear(in_features, n_actions))
        actor.append(torch.nn.Softmax(dim=-1))

        actor.to(self.device)
        shared_conv.to(self.device)
        critic.to(self.device)

        return shared_conv, actor, critic
