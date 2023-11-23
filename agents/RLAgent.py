import os
import time
from typing import Callable, Tuple
from agents.RLAgentParams import RLAgentParams
from agents.ReplayBuffer import ReplayBatch, ReplayBuffer


import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


class RLAgent(nn.Module):
    """
    Base class for all RL agents
    Usese exponential decay for epsilon
    """

    def __init__(self, args: RLAgentParams) -> None:
        super().__init__()
        self.args = args

        self.max_episodes = args.max_episodes
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.seed = args.seed
        self.device = args.device

        self.env = args.make_env()
        self.random_state = np.random.RandomState(args.seed)
        self.replay_buffer = ReplayBuffer(args.buffer_size, self.random_state)
        self.writer = SummaryWriter(
            log_dir=args.save_dir) if args.log else None

    def train(self, callback: Callable[[int], None]) -> None:
        """
        Trains the agent using the parameters provided

        Args:
            callback: a function to call after each episode
        """
        # exponential decay for epsilon
        epsilon = np.maximum(self.epsilon * 0.999 **
                             np.arange(self.max_episodes), 0.01)

        total_steps = 0
        for i, eps in enumerate(epsilon):

            episode_reward, episode_length = self.run_episode(
                total_steps, i, eps)

            self.log("reward", episode_reward, i)
            self.log("episode_length", episode_length, i)
            self.log("epsilon", eps, i)

            self.end_train_episode(i, total_steps)
            callback(i)

    def run_episode(self, total_steps, i, eps) -> Tuple[int, int]:
        """
        Runs a single episode of training

        Args:
            total_steps: the total number of steps taken
            i: the current episode
            eps: the current epsilon

        Returns:
            the episode reward and length
        """

        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            if self.random_state.rand() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(state)

            next_state, reward, done, *_ = self.env.step(action)

            episode_reward += reward
            episode_length += 1
            total_steps += 1
            self.replay_buffer.append(
                (state, action, reward, next_state, done))

            state = next_state

            if len(self.replay_buffer) >= self.batch_size:
                transitions = self.replay_buffer.draw(self.batch_size)
                self.train_model(transitions, i)

        return episode_reward, episode_length

    def end_train_episode(self, episode: int, total_steps: int) -> None:
        """
        Optional callback at the end of each episode

        Args:
            episode: the current episode
            total_steps: the total number of steps taken
        """
        pass

    def log(self, name: str, value: any, episode: int) -> None:
        """
        Logs a value to tensorboard

        Args:
            name: the name of the value
            value: the value to log
            episode: the current episode
        """
        if self.writer is not None:
            self.writer.add_scalar(name, value, episode)

    def get_action(self, state: np.ndarray):
        """
        Returns the action to take given the state
        Must be implemented by subclasses

        Args:
            state: the current state

        Returns:
            the action to take
        """
        raise NotImplementedError()

    def train_model(self, transitions: ReplayBatch, episode: int) -> None:
        """
        Trains the model using the transitions
        Must be implemented by subclasses

        Args:
            transitions: the transition batch to use for training
            episode: the current episode

        """
        raise NotImplementedError()

    def save(self, dir_path: str) -> None:
        """
        Saves the model and parameters to the given directory

        Args:
            dir_path: the directory to save to
        """
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.state_dict(), f"{dir_path}/model.pt")
        self.args.save_json_file(f"{dir_path}/params.json")

    @staticmethod
    def load(dir_path: str) -> "RLAgent":
        """
        Loads the model and parameters from the given directory

        Args:
            dir_path: the directory to load from
        """
        args = RLAgentParams()
        args.load_json_file(f"{dir_path}/params.json")

        agent = RLAgent(args)
        agent.load_state_dict(torch.load(f"{dir_path}/model.pt"))

        return agent

    def to_torch_batch(self, state: np.ndarray) -> torch.Tensor:
        """
        Converts an observation to a torch tensor to be used as input to models

        Args:
            state: the observation to convert

        Returns:
            a torch tensor (1, *state.shape)
        """
        return torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)

    def transition_to_torch_batch(self, transitions: ReplayBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """"
        Converts a ReplayBatch to a torch torch tensor to be used as input to models

        Args:
            transitions: the ReplayBatch to convert

        Returns:
            a tuple of torch tensors (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = transitions
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones

    def visualize(self, seed: int, max_length: int) -> None:
        """
        Visualizes the agent in the environment for 1 episode

        Args:
            seed: the random seed
            max_length: the maximum number of steps to take
        """

        env = self.args.make_env()
        vis_env = self.args.make_test_env()

        state, _ = env.reset(seed=seed)
        vis_env.reset(seed=seed)
        done = False
        total_length = 0
        total_reward = 0

        while not done:
            env.render()

            action = self.get_action(state)
            state, reward, done, *_ = env.step(action)
            total_length += 1
            total_reward += reward
            if total_length > max_length:
                break
            vis_env.step(action)
            time.sleep(0.1)

        print(f"Total length: {total_length}")
        print(f"Total reward: {total_reward}")
