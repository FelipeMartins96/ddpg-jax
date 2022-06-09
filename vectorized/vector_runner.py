import gym
import rsoccer_gym.experimental

envs = gym.vector.make("msc-v230", num_envs=3, asynchronous=True)

envs.reset()

import time
s = time.time()
for i in range(1000):
    envs.step(envs.action_space.sample())
