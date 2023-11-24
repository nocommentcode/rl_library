import argparse
import json
import time

import gym

from envs.gym_env_factory import gym_env_factory


class RLAgentParams:
    """
    Base class for RL agent parameters
    """
    # the json file path to load parameters from
    file: str
    # the name of the environment to train on
    env_name: str
    # the maximum number of training episodes
    max_episodes: int
    # the learning rate
    lr: float
    # the discount factor
    gamma: float
    # the initial exploration rate, will decay to 0.01
    epsilon: float
    # the batch size
    batch_size: int
    # the replay buffer size
    buffer_size: int
    # the random seed
    seed: int
    # whether to log to tensorboard
    log: bool
    # the device to run on
    device: str
    # the name of the run
    run_name: str
    # the name of the agent
    agent_name = "RLAgent"
    # train model after n steps
    model_train_steps: int
    # save directory
    save_dir: str

    def parse_args(self, args) -> None:
        if args.file is not None:
            self.load_json_file(args.file)
        else:
            arg_dict = vars(args)
            self.set_args(**arg_dict)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('-f', '--file', type=str, default=None,
                            help='JSON file path to load parameters from')
        parser.add_argument('-n', '--env_name', type=str, required=True,
                            help='Environment name')
        parser.add_argument('-e', '--max_episodes', type=int, default=1000,
                            help='Maximum number of training episodes')
        parser.add_argument('-l', '--lr', type=float, default=0.001,
                            help='Learning rate')
        parser.add_argument('-g', '--gamma', type=float, default=0.99,
                            help='Discount factor')
        parser.add_argument('-p', '--epsilon', type=float, default=0.1,
                            help='Initial Exploration rate, will decay to 0')
        parser.add_argument('-b', '--batch_size', type=int, default=32,
                            help='Batch size')
        parser.add_argument('-s', '--buffer_size', type=int, default=10000,
                            help='Replay buffer size')
        parser.add_argument('--seed', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--log', action='store_true',
                            help='Log to tensorboard')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to run on')
        parser.add_argument('-mt', '--model_train_steps', type=int, default=1,
                            help='Train model after n steps')
        parser.add_argument('--save_dir', type=str, default='runs',
                            help='Save directory')

    def load_json_file(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            args = json.load(f)
            print(args)
            self.set_args(**args)

    def save_json_file(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(vars(self), f)

    def set_args(self, **kwargs) -> None:
        self.env_name = kwargs['env_name']
        self.max_episodes = kwargs['max_episodes']
        self.lr = kwargs['lr']
        self.gamma = kwargs['gamma']
        self.epsilon = kwargs['epsilon']
        self.batch_size = kwargs['batch_size']
        self.buffer_size = kwargs['buffer_size']
        self.seed = kwargs['seed']
        self.log = kwargs['log']
        self.device = kwargs['device']
        self.model_train_steps = kwargs['model_train_steps']
        self.save_dir = kwargs['save_dir']
        self.run_name = f"{self.env_name}_{self.agent_name}_{time.strftime('%Y%m%d-%H%M%S')}"

    def make_env(self) -> gym.Env:
        return gym_env_factory(self.env_name)

    def make_test_env(self) -> gym.Env:
        return gym.make(self.env_name, render_mode='human')

    def __str__(self) -> str:
        return "".join([f"{k}: {str(v)}\n" for k, v in vars(self).items()])
