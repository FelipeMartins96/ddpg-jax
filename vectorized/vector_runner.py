import gym
from gym.wrappers import RecordVideo, VectorListInfo, RecordEpisodeStatistics
import rsoccer_gym.experimental
from agent import DDPG
from buffer import ReplayBuffer
import wandb
import numpy as np
from tqdm import tqdm
from flax.training import checkpoints

ENVIRONMENT_STEPS= int(5e6)
MIN_BUFFER = 1000
N_ENVS = 3
VAL_FREQ = 1000

class Logger:
    def __init__(self, steps_in_epoch=1000):
        self.steps_in_epoch = steps_in_epoch
        self.total_steps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.reset_epoch()
    
    def reset_epoch(self):
        self.epoch_steps = 0
        self.epoch_episodes = 0
        self.epoch_rw_sum = 0
        self.epoch_updates = 0
        self.epoch_update_dict = {'pi_loss': 0, 'q_loss': 0}
        self.epoch_rsoccer_rws = np.zeros(5, dtype=np.float32)
        self.epoch_rsoccer_raw_rws = np.zeros(5, dtype=np.float32)

    def add_update_info(self, pi_loss, q_loss):
        self.epoch_update_dict['pi_loss'] += pi_loss
        self.epoch_update_dict['q_loss'] += q_loss
        self.epoch_updates += 1

    def add_ep_infos(self, info):
        if 'rewards' in info:
            self.epoch_rsoccer_rws += info['rewards']
            self.epoch_rsoccer_raw_rws += info['raw_rewards']
        
        self.epoch_steps += info['episode']['l']
        self.epoch_rw_sum += info['episode']['r']
        self.epoch_episodes += 1

        if self.epoch_steps > self.steps_in_epoch:
            self.log()

    def log(self):
        self.total_steps += self.epoch_steps
        self.total_episodes += self.epoch_episodes
        self.total_updates += self.epoch_updates

        rsoccer_rws = self.epoch_rsoccer_rws / self.epoch_episodes
        rsoccer_raw_rws = self.epoch_rsoccer_raw_rws / self.epoch_episodes
        log = {
            'episode/steps': self.epoch_steps / self.epoch_episodes,
            'episode/rewards': self.epoch_rw_sum / self.epoch_episodes,
            'rsoccer_rewards/m_w_goal': rsoccer_rws[0],
            'rsoccer_rewards/m_w_ball_grad': rsoccer_rws[1],
            'rsoccer_rewards/m_w_move': rsoccer_rws[2],
            'rsoccer_rewards/m_w_collision': rsoccer_rws[3],
            'rsoccer_rewards/m_w_energy': rsoccer_rws[4],
            'rsoccer_raw_rewards/m_w_goal': rsoccer_raw_rws[0],
            'rsoccer_raw_rewards/m_w_ball_grad': rsoccer_raw_rws[1],
            'rsoccer_raw_rewards/m_w_move': rsoccer_raw_rws[2],
            'rsoccer_raw_rewards/m_w_collision': rsoccer_raw_rws[3],
            'rsoccer_raw_rewards/m_w_energy': rsoccer_raw_rws[4],
        }
        if self.epoch_updates > 0:
            log.update({
            'agent/pi_loss': self.epoch_update_dict['pi_loss'] / self.epoch_updates,
            'agent/q_loss': self.epoch_update_dict['q_loss'] / self.epoch_updates,})
        wandb.log(log, step=int(self.total_steps))

        self.reset_epoch()


def run_validation_ep(agent, env):
    obs = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        _obs, _, done, _ = env.step(action)
        obs = _obs


if __name__ == '__main__':
    wandb.init(
        project='msc-w23',
        entity='felipemartins',
        monitor_gym=True,
    )

    envs = gym.vector.make("Pendulum-v1", num_envs=N_ENVS, asynchronous=True)
    envs = RecordEpisodeStatistics(envs)
    envs = VectorListInfo(envs)
    val_env = gym.make("Pendulum-v1")
    val_env = RecordVideo(
        val_env, video_folder="gym_recordings", episode_trigger=lambda x: True
    )

    agent = DDPG(
        obs_space=envs.single_observation_space,
        act_space=envs.single_action_space,
        lr_c=2e-4,
        lr_a=1e-4,
        gamma=0.95,
        seed=0,
        sigma=0.2,
    )
    buffer = ReplayBuffer(
        env_observation_space=envs.single_observation_space,
        env_action_space=envs.single_action_space,
        capacity=ENVIRONMENT_STEPS,
    )
    logger = Logger()

    obs = envs.reset()
    for step in tqdm(range(int(ENVIRONMENT_STEPS/N_ENVS)), smoothing=0.01):
        buffering = buffer.size < MIN_BUFFER

        if VAL_FREQ and not buffering and step % VAL_FREQ == 0:
            run_validation_ep(agent, val_env)
            checkpoints.save_checkpoint(
                f'./checkpoints/',
                agent.actor_params,
                step=step,
                overwrite=True,
            )

        if not buffering:
            actions = np.array(agent.sample_action(obs))
        else:
            actions = envs.action_space.sample()

        n_obs, rewards, dones, infos = envs.step(actions)
        for i in range(N_ENVS):
            terminal_state = dones[i]
            _obs = n_obs[i]
            if dones[i]:
                logger.add_ep_infos(infos[i])
                if "TimeLimit.truncated" in infos[i]:
                    terminal_state = False
                    _obs = infos[i]['terminal_observation']
                
            buffer.add(obs[i], actions[i], rewards[i], terminal_state, _obs)

        if buffer.size >= MIN_BUFFER:
            batch = buffer.get_batch(64)
            pi_loss, q_loss = agent.update(batch)
            logger.add_update_info(pi_loss, q_loss)

        obs = n_obs

    checkpoints.save_checkpoint(
        f'./checkpoints/',
        agent.actor_params,
        step=int(ENVIRONMENT_STEPS/N_ENVS),
        overwrite=True,
    )