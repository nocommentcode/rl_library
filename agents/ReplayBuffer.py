

from typing import Tuple
import numpy as np


from collections import deque

ReplayBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """
    A simple replay buffer for storing transitions
    """

    def __init__(self, buffer_size: int, random_state: np.random.RandomState) -> None:
        """
        Initializes the replay buffer

        Args:
            buffer_size: the maximum number of transitions to store
            random_state: a random state for sampling
        """

        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, transition) -> None:
        self.buffer.append(transition)

    def draw(self, batch_size: int) -> ReplayBatch:
        indices = self.random_state.choice(len(self.buffer), batch_size,
                                           replace=False)

        states = np.array([self.buffer[index][0] for index in indices])
        actions = np.array([self.buffer[index][1] for index in indices])
        rewards = np.array([self.buffer[index][2] for index in indices])
        next_states = np.array([self.buffer[index][3] for index in indices])
        dones = np.array([self.buffer[index][4] for index in indices])

        return states, actions, rewards, next_states, dones
