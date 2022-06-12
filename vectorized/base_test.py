import gym
import rsoccer_gym.experimental
from gym.wrappers import VectorListInfo, RecordEpisodeStatistics

if __name__ == '__main__':
    envs = gym.vector.make("Pendulum-v1", num_envs=3, asynchronous=False)
    envs = RecordEpisodeStatistics(envs)
    envs = VectorListInfo(envs)

    envs.reset()

    while True:
        actions = envs.action_space.sample()
        n_obs, rewards, dones, infos = envs.step(actions)
        for i in range(3):
            # terminal_state = False if not dones[i] or "TimeLimit.truncated" in infos[i] else True
            if dones[i]:
                print(infos[i])
