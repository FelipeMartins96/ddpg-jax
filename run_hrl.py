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
        'manager/goal': info['manager_weighted_rw'][0],
        'manager/ball_grad': info['manager_weighted_rw'][1],
        # 'manager/move': info['manager_weighted_rw'][2],
        # 'manager/collision': info['manager_weighted_rw'][3],
        # 'manager/energy': info['manager_weighted_rw'][4],
        'worker/dist': info['workers_weighted_rw'][0][0],
        # 'worker/energy': info['workers_weighted_rw'][0][1],
    }


def run_validation_ep(m_agent, w_agent, env, opponent_policies):
    m_obs = env.reset()
    done = False
    while not done:
        m_action = m_agent.get_action(m_obs)
        w_obs = env.set_action_m(m_action)
        w_action = w_agent.get_action(w_obs)
        step_action = np.concatenate(
            [w_action] + [[p()] for p in opponent_policies], axis=0
        )
        _obs, _, done, _ = env.step(step_action)
        m_obs = _obs.manager


def main(args):
    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        monitor_gym=args.wandb_monitor_gym,
        config=args,
    )
    total_training_steps = args.training_total_steps + args.training_replay_min_size
    replay_capacity = args.training_total_steps + args.training_replay_min_size
    min_replay_size = args.training_replay_min_size
    batch_size = args.training_batch_size
    gamma = args.training_gamma
    learning_rate = args.training_learning_rate
    seed = args.seed

    env = gym.make(
        args.env_name,
        n_robots_blue=args.env_n_robots_blue,
        n_robots_yellow=args.env_n_robots_yellow,
    )
    if args.training_val_frequency:
        val_env = gym.wrappers.RecordVideo(
            gym.make(
                args.env_name,
                n_robots_blue=args.env_n_robots_blue,
                n_robots_yellow=args.env_n_robots_yellow,
            ),
            './monitor/',
            episode_trigger=lambda x: True,
        )
    key = jax.random.PRNGKey(seed)
    if args.env_opponent_policy == 'off':
        opponent_policies = [
            lambda: np.array([0.0, 0.0]) for _ in range(args.env_n_robots_yellow)
        ]
    env.set_key(key)
    val_env.set_key(key) if args.training_val_frequency else None

    m_observation_space, m_action_space = env.get_spaces_m()
    w_observation_space, w_action_space = env.get_spaces_w()

    m_agent = DDPG(m_observation_space, m_action_space, learning_rate, gamma, seed)
    w_agent = DDPG(w_observation_space, w_action_space, learning_rate, gamma, seed)
    w_agent.actor_params = checkpoints.restore_checkpoint(
        './checkpoints/pretraining-worker', w_agent.actor_params
    )
    m_buffer = ReplayBuffer(m_observation_space, m_action_space, replay_capacity)
    w_buffer = ReplayBuffer(w_observation_space, w_action_space, replay_capacity)

    m_obs = env.reset()
    rewards, ep_steps, done  = 0, 0, False
    m_q_losses, m_pi_losses, w_q_losses, w_pi_losses = [], [], [], []
    for step in tqdm(range(total_training_steps), smoothing=0.01):
        if args.training_val_frequency and step % args.training_val_frequency == 0:
            run_validation_ep(m_agent, w_agent, val_env, opponent_policies)

        m_action = np.array(m_agent.sample_action(m_obs))
        w_obs = env.set_action_m(m_action)
        w_actions = np.array(w_agent.sample_action(w_obs))
        step_action = np.concatenate(
            [w_actions] + [[p()] for p in opponent_policies], axis=0
        )
        _obs, reward, done, info = env.step(step_action)

        terminal_state = False if not done or "TimeLimit.truncated" in info else True

        m_buffer.add(m_obs, m_action, 0.0, reward.manager, terminal_state, _obs.manager)
        for i in range(args.env_n_robots_blue):
            w_buffer.add(w_obs[i], w_actions[i], 0.0, reward.workers[i], terminal_state, _obs.workers[i])

        rewards += reward.manager
        ep_steps += 1
        if step >= min_replay_size:
            m_batch = m_buffer.get_batch(batch_size)
            m_act_loss, m_crt_loss = m_agent.update(m_batch)
            m_pi_losses.append(m_act_loss)
            m_q_losses.append(m_crt_loss)

            if step < 1000000 or step % 10 == 0:
                w_batch = w_buffer.get_batch(batch_size)
                w_act_loss, w_crt_loss = w_agent.update(w_batch)
                w_pi_losses.append(w_act_loss)
                w_q_losses.append(w_crt_loss)

            if done:
                log = info_to_log(info)
                log.update(
                    {
                        'ep_reward': rewards,
                        'ep_steps': ep_steps,
                        'manager/q_loss': sum(m_q_losses) / len(m_q_losses),
                        'manager/pi_loss': sum(m_pi_losses) / len(m_pi_losses),
                        'worker/q_loss': sum(w_q_losses) / len(w_q_losses),
                        'worker/pi_loss': sum(w_pi_losses) / len(w_pi_losses),
                    }
                )
                wandb.log(log, step=step)
                m_q_losses, m_pi_losses, w_q_losses, w_pi_losses = [], [], [], []

        m_obs = _obs.manager
        if done:
            m_obs = env.reset()
            rewards, ep_steps = 0, 0

    checkpoints.save_checkpoint(
        f'./checkpoints/m_{args.env_name}/{args.wandb_name}',
        m_agent.actor_params,
        step=total_training_steps,
        overwrite=True,
    )
    checkpoints.save_checkpoint(
        f'./checkpoints/w_{args.env_name}/{args.wandb_name}',
        w_agent.actor_params,
        step=total_training_steps,
        overwrite=True,
    )


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
    parser.add_argument('--env-name', type=str, default='VSSHRL-v0')
    parser.add_argument('--env-n-robots-blue', type=int, default=1)
    parser.add_argument('--env-n-robots-yellow', type=int, default=0)
    parser.add_argument('--env-opponent-policy', type=str, default='off')

    # TRAINING
    parser.add_argument('--training-total-steps', type=int, default=5000000)
    parser.add_argument('--training-replay-min-size', type=int, default=100000)
    parser.add_argument('--training-batch-size', type=int, default=256)
    parser.add_argument('--training-gamma', type=float, default=0.95)
    parser.add_argument('--training-learning-rate', type=float, default=1e-4)
    parser.add_argument('--training-val-frequency', type=int, default=250000)
    parser.add_argument('--training-load-worker', type=bool, default=True)

    args = parser.parse_args()
    main(args)
