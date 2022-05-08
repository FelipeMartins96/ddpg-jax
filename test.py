import time
from argparse import ArgumentParser
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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

# TODO usar random de jax no validation loop
# TODO Criar ambiente
# TODO Criar agents
# TODO loop de validation ep


def run_validation_ep(m_agent, w_agent, env, n_controlled_robots, key):
    m_obs = env.reset()
    # env.render()
    done = False
    ep_rw = 0
    ep_steps = 0

    @jax.jit()
    def random_m_action(k):
        k1, k2 = jax.random.split(k, 2)
        return k1, jax.random.uniform(k2,minval=-1, maxval=1, shape=(2, 2*n_controlled_robots))

    @jax.jit()
    def random_w_action(k):
        k1, k2 = jax.random.split(k, 2)
        return k1, jax.random.uniform(k2,minval=-1, maxval=1, shape=(2*3, 2))
    key, n_k = jax.random.split(key)

    while not done:
        key, m_action = random_m_action(key)
        w_obs = env.set_action_m(np.array(m_action))
        key, w_action = random_w_action(key)
        _obs, rw, done, info = env.step(np.array(w_action))
        ep_rw += rw.manager[0]
        ep_steps += 1
        m_obs = _obs.manager

        # env.render()
    return key

w_checkpoint = './checkpoints/VSSHRL-v1/pretraining-worker'
n_controlled_robots = 1
seed = 0

env = gym.make('VSSHRLSelf-v0', n_controlled_robots=n_controlled_robots)

key = jax.random.PRNGKey(seed)
env.set_key(key)

m_observation_space, m_action_space = env.get_spaces_m()
w_observation_space, w_action_space = env.get_spaces_w()

m_agent = DDPG(
    m_observation_space, m_action_space, 0.1, 0.1, 0.9, seed, 0.2
)
w_agent = DDPG(
    w_observation_space, w_action_space, 0.1, 0.1, 0.9, seed, 0.2
)
if w_checkpoint:
    w_agent.actor_params = checkpoints.restore_checkpoint(
        w_checkpoint, w_agent.actor_params
    )

done = False

import time
a = time.time()
key = run_validation_ep(m_agent, w_agent, env, n_controlled_robots, key)
print(time.time()-a)