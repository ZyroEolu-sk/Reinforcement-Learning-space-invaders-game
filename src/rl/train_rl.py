from gym_env import SpaceInvadersGymEnv

from stable_baselines3 import Monitor

def make_train_env():
    return Monitor(SpaceInvadersGymEnv(render_mode=None, max_steps=4500, frame_skip=2))

def make_eval_env():
    return Monitor(SpaceInvadersGymEnv(render_mode=None, max_steps=4500, frame_skip=2))