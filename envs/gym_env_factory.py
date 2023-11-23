import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from envs.ClassicControlWrapper import ClassicControlWrapper
from gym.wrappers.frame_stack import FrameStack

ATARI_ENVS = [
    "ALE/Breakout-v5"
]

CLASSIC_CONTROL_ENVS = [
    "CartPole-v0"
]


def gym_env_factory(name: str) -> gym.Env:
    """
    Create a wrapped gym environment with the given name.
    """
    if name in ATARI_ENVS:
        env = AtariPreprocessing(
            gym.make(name, frameskip=1, full_action_space=False), scale_obs=True)

    if name in CLASSIC_CONTROL_ENVS:
        env = ClassicControlWrapper(gym.make(name, render_mode='rgb_array'))

    return FrameStack(env, num_stack=4)
