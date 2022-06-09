import gym
import rsoccer_gym.experimental



envs = gym.make("msc-v230")

envs.reset()

actions = envs.action_space.sample()
envs.step(actions)