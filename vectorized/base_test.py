import gym
import rsoccer_gym.experimental

envs = gym.vector.make("msc-v230", num_envs=3, asynchronous=True)

envs.reset()

actions = envs.action_space.sample()

print(envs.action_space)
# observations, rewards, dones, infos = envs.step(actions)
print(envs.step(actions))