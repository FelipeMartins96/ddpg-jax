import time
from argparse import ArgumentParser

import gym
import jax
import numpy as np
import pyvirtualdisplay
import rsoccer_gym
import wandb
from flax.training import checkpoints
from tqdm import tqdm

from agent import DDPG
from buffer import ReplayBuffer


def info_to_log(info):
    return {
        'manager/goal': info['manager_weighted_rw'][0],
        'manager/ball_grad': info['manager_weighted_rw'][1],
        'manager/move': info['manager_weighted_rw'][2],
        # 'manager/collision': info['manager_weighted_rw'][3],
        'manager/energy': info['manager_weighted_rw'][4],
        # 'worker/dist': info['workers_weighted_rw'][0][0],
        # 'worker/energy': info['workers_weighted_rw'][0][1],
    }


def run_validation_ep(agent, env, opponent_policies):
    obs = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        action = np.array(action.reshape((-1, 2)))
        step_action = np.concatenate(
            [action] + [[p()] for p in opponent_policies], axis=0
        )
        _obs, _, done, _ = env.step(step_action)
        obs = _obs.manager


def main():
    n_robots_blue = 1
    n_robots_yellow = 1

    env = gym.make(
        'VSSHRL-v3',
        n_robots_blue=n_robots_blue,
        n_robots_yellow=n_robots_yellow,
        hierarchical=False,
    )
    seed = 0
    key = jax.random.PRNGKey(seed)
    opponent_policies = [
        lambda: np.array([0.0, 0.0]) for _ in range(n_robots_yellow)
    ]
    env.set_key(key)
    m_observation_space, m_action_space = env.get_spaces_m()
    w_observation_space, w_action_space = env.get_spaces_w()
    w_action_space = gym.spaces.Box(
        low=-1, high=1, shape=((n_robots_blue) * 2,), dtype=np.float32
    )

    # agent = DDPG(m_observation_space, w_action_space, 0.0, 0.0, seed)

    obs = env.reset()
    done = False
    while not done:
        env.render()
        input()
        # action = np.array(agent.sample_action(obs))
        action = np.array([1.,1.])
        # print(action)
        step_action = np.concatenate(
            [action.reshape((-1, 2))] + [[p()] for p in opponent_policies], axis=0
        )
        _obs, reward, done, info = env.step(step_action)
        print(step_action)
        print(info_to_log(info))
        obs = _obs.manager


if __name__ == '__main__':
    main()
