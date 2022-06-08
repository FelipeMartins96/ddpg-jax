import rsoccer_gym.experimental
import gym
import jax
import numpy as np
# from vector_env import AsyncHRLVectorEnv, _worker_shared_memory
from gym.vector.async_vector_env import AsyncVectorEnv

env_fns = [lambda: gym.make("Pendulum-v1")] * 5
envs = AsyncVectorEnv(env_fns)
envs.seed(0)
envs.reset()
done = np.array([False, False])

while not done.any():
    w_action = envs.action_space.sample()
    print(envs.action_space.sample())
    # w_action = np.random.uniform(-1, 1, size=(6, 2))
    _obs, reward, done, _ = envs.step(w_action)
input()
# m_obs = _obs.manager
