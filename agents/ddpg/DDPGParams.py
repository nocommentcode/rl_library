from agents.FeatureEncoderParams import FeatureEncoderParams

import torch
import torch.nn as nn

import argparse
from typing import List, Tuple


class DDPGParams(FeatureEncoderParams):
    actor_fc: List[int]
    critic_fc: List[int]
    phi: float
    exploration_sigma: float

    agent_name = "DDPG"

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
        parser.add_argument('-phi', type=float,
                            default=0.95,
                            help='Target update rate')
        parser.add_argument('-es', '--exploration_sigma', type=float,
                            default=0.2,
                            help='Exploration sigma')

    def set_args(self, **kwargs) -> None:
        super().set_args(**kwargs)
        self.actor_fc = kwargs['actor_fully_connected'] if 'actor_fully_connected' in kwargs \
            else kwargs['afc']
        self.critic_fc = kwargs['critic_fully_connected'] if 'critic_fully_connected' in kwargs \
            else kwargs['cfc']
        self.phi = kwargs['phi']
        self.exploration_sigma = kwargs['exploration_sigma'] if 'exploration_sigma' in kwargs \
            else kwargs['es']

    def build_model(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Builds the model based on the parameters

        Returns:
            shared_conv: The shared convolutional layers
            actor: The actor network
            critic: The critic network
        """
        env = self.make_env()
        observation_space = env.observation_space
        state_space = observation_space.shape
        n_actions = env.action_space.shape[0]

        shared_conv = self.build_encoder()

        conv_output_features = shared_conv(
            torch.zeros(1, *state_space).to(self.device)).shape[1]

        # critic
        critic = nn.Sequential()
        # ddpg gets the state and action as input to the critic
        in_features = conv_output_features + n_actions
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

        # ddpg gets a tanh activation on the output
        actor.append(torch.nn.Tanh())

        actor.to(self.device)
        shared_conv.to(self.device)
        critic.to(self.device)

        return shared_conv, actor, critic
