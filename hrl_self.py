import time
from argparse import ArgumentParser

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


def info_to_log(info):
    return {
        'ep_info/m_blue_rw': info['blue_manager_weighted_rw'].sum(),
        'ep_info/m_blue_goal': info['blue_manager_weighted_rw'][0],
        'ep_info/m_blue_ball_grad': info['blue_manager_weighted_rw'][1],
        'ep_info/w_mean_dist': info['workers_weighted_rw'].mean(axis=0)[0],
    }


def run_validation_ep(m_agent, w_agent, env):
    m_obs = env.reset()
    done = False
    ep_rw = 0
    ep_steps = 0
    while not done:
        m_action = m_agent.get_action(m_obs[0])
        m_action = np.concatenate([m_action, [2.0] * 6])
        w_obs = env.set_action_m(m_action)
        w_action = w_agent.get_action(w_obs[3:])
        w_action = np.concatenate([w_action, np.zeros((3, 2))])
        _obs, rw, done, info = env.step(w_action)
        ep_rw += rw.manager[0]
        ep_steps += 1
        m_obs = _obs.manager
    wandb.log(
        {
            'validation/ep_steps': ep_steps,
            'validation/m_blue_rw': info['blue_manager_weighted_rw'].sum(),
            'validation/m_blue_goal': info['blue_manager_weighted_rw'][0],
            'validation/m_blue_ball_grad': info['blue_manager_weighted_rw'][1],
            'validation/w_mean_dist': info['workers_weighted_rw'][:3].mean(axis=0)[0],
        },
        commit=False,
    )


def main(args):
    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        monitor_gym=args.wandb_monitor_gym,
        config=args,
    )
    min_replay_size = args.training_replay_min_size
    total_training_steps = min_replay_size + (
        args.training_grad_steps * args.training_steps_grad_ratio
    )
    step_ratio = args.training_steps_grad_ratio
    replay_capacity = total_training_steps
    batch_size = args.training_batch_size
    lr_critic = args.training_lr_critic
    lr_actor = args.training_lr_actor
    gamma_m = args.training_gamma_manager
    gamma_w = args.training_gamma_worker
    seed = args.seed
    val_frequency = args.training_val_frequency * step_ratio

    env = gym.make(args.env_name)
    val_env = (
        gym.wrappers.RecordVideo(
            gym.make(args.env_name), './monitor/', episode_trigger=lambda x: True
        )
        if args.training_val_frequency
        else None
    )

    key = jax.random.PRNGKey(seed)
    env.set_key(key)
    val_env.set_key(key) if val_env else None

    m_observation_space, m_action_space = env.get_spaces_m()
    w_observation_space, w_action_space = env.get_spaces_w()

    m_agent = DDPG(
        m_observation_space, m_action_space, lr_critic, lr_actor, gamma_m, seed
    )
    w_agent = DDPG(
        w_observation_space, w_action_space, lr_critic, lr_actor, gamma_w, seed
    )

    m_buffer = ReplayBuffer(m_observation_space, m_action_space, replay_capacity)
    w_buffer = ReplayBuffer(w_observation_space, w_action_space, replay_capacity)

    @jax.jit
    def random_m_action(k):
        k1, k2 = jax.random.split(k, 2)
        return k1, jax.random.uniform(k2, shape=(2, 6))

    @jax.jit
    def random_w_action(k):
        k1, k2 = jax.random.split(k, 2)
        return k1, jax.random.uniform(k2, shape=(6, 2))

    m_obs = env.reset()
    ep_steps = 0
    m_q_losses, m_pi_losses, w_q_losses, w_pi_losses = [], [], [], []
    for step in tqdm(range(total_training_steps), smoothing=0.01):
        buffering = step < min_replay_size

        if val_frequency and not buffering and step % val_frequency == 0:
            run_validation_ep(m_agent, w_agent, val_env)
            checkpoints.save_checkpoint(
                f'./checkpoints/m_{args.env_name}/{args.wandb_name}',
                m_agent.actor_params,
                step=step,
                overwrite=True,
            )
            checkpoints.save_checkpoint(
                f'./checkpoints/w_{args.env_name}/{args.wandb_name}',
                w_agent.actor_params,
                step=step,
                overwrite=True,
            )

        if buffering:
            key, m_action = random_m_action(key)
            m_action = np.array(m_action)
            w_obs = env.set_action_m(m_action)
            key, w_actions = random_w_action(key)
            w_actions = np.array(w_actions)
        else:
            m_action = np.array(m_agent.sample_action(m_obs))
            w_obs = env.set_action_m(m_action)
            w_actions = np.array(w_agent.sample_action(w_obs))

        _obs, rws, done, info = env.step(w_actions)

        ts = False if not done or "TimeLimit.truncated" in info else True

        for i in range(2):
            m_buffer.add(m_obs[i], m_action[i], rws.manager[i], ts, _obs.manager[i])
        for i in range(6):
            w_buffer.add(w_obs[i], w_actions[i], rws.workers[i], ts, _obs.workers[i])

        ep_steps += 1

        if not buffering:
            pi_loss, q_loss = m_agent.update(m_buffer.get_batch(batch_size))
            m_pi_losses.append(pi_loss), m_q_losses.append(q_loss)
            if step % 10 == 0:
                pi_loss, q_loss = w_agent.update(w_buffer.get_batch(batch_size))
                w_pi_losses.append(pi_loss), w_q_losses.append(q_loss)

            if done:
                log = info_to_log(info)
                log.update(
                    {
                        'ep_info/ep_steps': ep_steps,
                        'train_info/manager_q_loss': np.mean(m_q_losses),
                        'train_info/manager_pi_loss': np.mean(m_pi_losses),
                        'train_info/worker_q_loss': np.mean(w_q_losses),
                        'train_info/worker_pi_loss': np.mean(w_pi_losses),
                    }
                )
                wandb.log(log, step=step)
                m_q_losses, m_pi_losses, w_q_losses, w_pi_losses = [], [], [], []

        m_obs = _obs.manager
        if done:
            m_obs, ep_steps = env.reset(), 0




if __name__ == '__main__':
    # Creates a virtual display for OpenAI gym
    pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    parser = ArgumentParser(fromfile_prefix_chars='@')
    # RANDOM
    parser.add_argument('--seed', type=int, default=0)

    # EXPERIMENT
    parser.add_argument('--experiment-type', type=str, default='not-set')

    # WANDB
    parser.add_argument('--wandb-mode', type=str, default='disabled')
    parser.add_argument('--wandb-project', type=str, default='rsoccer-hrl')
    parser.add_argument('--wandb-entity', type=str, default='felipemartins')
    parser.add_argument('--wandb-name', type=str)
    parser.add_argument('--wandb-monitor-gym', type=bool, default=True)

    # ENVIRONMENT
    parser.add_argument('--env-name', type=str, default='VSSHRLSelf-v0')

    # TRAINING
    parser.add_argument('--training-grad-steps', type=int, default=5000000)
    parser.add_argument('--training-replay-min-size', type=int, default=250000)
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--training-gamma-manager', type=float, default=0.95)
    parser.add_argument('--training-gamma-worker', type=float, default=0.95)
    parser.add_argument('--training-lr-critic', type=float, default=2e-4)
    parser.add_argument('--training-lr-actor', type=float, default=1e-4)
    parser.add_argument('--training-val-frequency', type=int, default=250000)
    parser.add_argument('--training-steps-grad-ratio', type=int, default=10)

    args = parser.parse_args()
    main(args)
