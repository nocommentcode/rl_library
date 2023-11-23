import cv2
import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.frame_stack import FrameStack


class ClassicControlWrapper(PixelObservationWrapper):
    """
    Wraps classic control environments to use pixel observations.
     - state is pixels
     - state is greyscale
     - state is scaled
     - state is resized
    """
    rgb_to_grey = np.array([0.299, 0.587, 0.114])

    def __init__(self, env: gym.Env, screen_size: int = 84):
        super().__init__(env, pixels_only=True)
        self.screen_size = screen_size
        shape = (screen_size, screen_size)
        dtype = np.float32
        self.observation_space = Box(low=np.zeros(shape, dtype=dtype),
                                     high=np.ones(shape, dtype=dtype),
                                     dtype=dtype,
                                     shape=shape)

    def observation(self, observation):
        pixels = super().observation(observation)['pixels']
        grey_scale = np.dot(pixels, self.rgb_to_grey)
        scaled = grey_scale / 255.0
        assert cv2 is not None
        resized = cv2.resize(
            scaled,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized
