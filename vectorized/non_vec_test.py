import gym
import rsoccer_gym.experimental
from gym.wrappers import RecordEpisodeStatistics



envs = gym.make("msc-v230")

envs.reset()
done = False
while not done:
    envs.render()
    actions = envs.action_space.sample()
    _, _, done, info = envs.step(actions)
