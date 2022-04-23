import rsoccer_gym
import gym
import jax
import numpy as np
from vector_env import AsyncHRLVectorEnv, _worker_shared_memory

env_fns = [lambda: gym.make("VSSHRLSelf-v0")] * 5
envs = AsyncHRLVectorEnv(env_fns, worker=_worker_shared_memory)
envs.seed(0)
envs.reset()
done = False

while not done:
    m_action = np.random.uniform(-1, 1, size=(2, 6))
    # w_obs = env.set_action_m(m_action)
    w_action = envs.action_space.sample()
    # w_action = np.random.uniform(-1, 1, size=(6, 2))
    _obs, reward, done, _ = envs.step(w_action)
input()
# m_obs = _obs.manager
