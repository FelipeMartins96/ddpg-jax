import gym
from gym.wrappers import RecordVideo, VectorListInfo, RecordEpisodeStatistics
import rsoccer_gym.experimental
from agent import DDPG
from buffer import ReplayBuffer
import wandb
import numpy as np
from tqdm import tqdm
from flax.training import checkpoints

VALIDATION_EPS= 100
ENV_NAME = 'msc-v251'

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
        self.sigma = 0.0

    def add_update_info(self, pi_loss, q_loss, sigma):
        self.epoch_update_dict['pi_loss'] += pi_loss
        self.epoch_update_dict['q_loss'] += q_loss
        self.sigma += sigma
        self.epoch_updates += 1

    def add_ep_infos(self, info, agent):
        self.epoch_rsoccer_rws += info[f'rewards-{agent}']
        
        self.epoch_steps += info['episode']['l']
        self.epoch_rw_sum += self.epoch_rsoccer_rws.sum()
        self.epoch_episodes += 1

        if self.epoch_steps > self.steps_in_epoch:
            self.log()

    def log(self):
        self.total_steps += self.epoch_steps
        self.total_episodes += self.epoch_episodes
        self.total_updates += self.epoch_updates

        rsoccer_rws = self.epoch_rsoccer_rws / self.epoch_episodes
        log = {
            'episode/epoch_episodes': self.epoch_episodes,
            'episode/steps': self.epoch_steps / self.epoch_episodes,
            'episode/rewards': self.epoch_rw_sum / self.epoch_episodes,
            'episode/total_episodes': self.total_episodes,
            'episode/total_updates': self.total_updates,
            'rsoccer_rewards/goal': rsoccer_rws[0],
            'rsoccer_rewards/ball_grad': rsoccer_rws[1],
            'rsoccer_rewards/move': rsoccer_rws[2],
            'rsoccer_rewards/collision': rsoccer_rws[3],
            'rsoccer_rewards/energy': rsoccer_rws[4],
        }
        if self.epoch_updates > 0:
            log.update({
            'agent/pi_loss': self.epoch_update_dict['pi_loss'] / self.epoch_updates,
            'agent/q_loss': self.epoch_update_dict['q_loss'] / self.epoch_updates,
            'agent/opponent_sigma': self.sigma / self.epoch_updates})
        wandb.log(log, step=int(self.total_steps))

        self.reset_epoch()


def run_validation_ep(agent, opponent, env):
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action = {}
        for agt in env.agents:
            if 'b' in agt:
                action[agt] = np.asarray(agent.get_action(obs[agt]))
            if 'y' in agt:
                action[agt] = np.asarray(opponent.get_action(obs[agt]))
        _obs, _, done, infos = env.step(action)
        steps += 1
        obs = _obs
    infos.update({'ep_steps': steps})
    return infos

if __name__ == '__main__':
    wandb.init(
        project='msc-w25',
        entity='felipemartins',
        monitor_gym=True,
    )

    val_env = gym.make(ENV_NAME)
    val_env = RecordVideo(
        val_env, video_folder="gym_recordings", episode_trigger=lambda x: True
    )

    agents = val_env.agents
    blue_agents = [a for a in agents if 'b'in a]
    
    agent = DDPG(
        obs_space=val_env.observation_space['b_0'],
        act_space=val_env.action_space['b_0'],
        lr_c=2e-4,
        lr_a=1e-4,
        gamma=0.95,
        seed=0,
        sigma=0.2,
    )

    opponent = DDPG(
        obs_space=val_env.observation_space['y_0'],
        act_space=val_env.action_space['y_0'],
        lr_c=2e-4,
        lr_a=1e-4,
        gamma=0.95,
        seed=0,
        sigma=0.7,
        theta=0.15,
        ou=False
    )
    opponent.actor_params = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints/msc-v250-pretrain', target=opponent.actor_params)
    agent.actor_params = checkpoints.restore_checkpoint(ckpt_dir='./checkpoints/msc-v251-dynamic', target=agent.actor_params)

    np.random.seed(0)
    rws = None
    ep_steps = 0
    goals_blue, goals_yellow, tie = 0, 0, 0
    for step in tqdm(range(int(VALIDATION_EPS)), smoothing=0.01):
        infos = run_validation_ep(agent, opponent, val_env)
        if rws is None:
            rws = infos['rewards-b_0']
        else:
            rws += infos['rewards-b_0']
        ep_steps += infos['ep_steps']
        goals_blue += 1 if infos['rewards-b_0'][0] == 1 else 0
        goals_yellow += 1 if infos['rewards-b_0'][0] == -1 else 0
        tie += 1 if infos['rewards-b_0'][0] == 0 else 0
    wandb.log({
        'validation/total_eps': VALIDATION_EPS,
        'validation/goal': rws[0] / VALIDATION_EPS,
        'validation/ball_grad': rws[1] / VALIDATION_EPS,
        'validation/move': rws[2] / VALIDATION_EPS,
        'validation/collision': rws[3] / VALIDATION_EPS,
        'validation/energy': rws[4] / VALIDATION_EPS,
        'validation/total_rw': np.sum(rws) / VALIDATION_EPS,
        'validation/ep_steps': ep_steps / VALIDATION_EPS,
        'validation/goals_blue': goals_blue / VALIDATION_EPS,
        'validation/goals_yellow': goals_yellow / VALIDATION_EPS,
        'validation/goals_tie': tie / VALIDATION_EPS,
        })
