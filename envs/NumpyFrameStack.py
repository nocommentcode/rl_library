import numpy as np
from gym.wrappers.frame_stack import FrameStack


class NumpyFrameStack(FrameStack):
    """
    Wrapper for gym environments that stacks observations along the first axis.

    """

    def observation(self, observation):
        return np.array(super().observation(observation))
