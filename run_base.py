import time
from argparse import ArgumentParser
from tqdm import tqdm
import gym
import numpy as np

from buffer import ReplayBuffer
from agent import DDPG

import jax
import wandb
import rsoccer_gym

def run_validation_ep(agent, env, opponent_policies):
    obs = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        step_action = np.concatenate([action] + [[p()] for p in opponent_policies], axis=0)
        _obs, _, done, _ = env.step(step_action)
        obs = _obs.manager

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
        hierarchical=False
    )
    if args.training_val_frequency:
        val_env = gym.wrappers.RecordVideo(
            gym.make(
                args.env_name,
                n_robots_blue=args.env_n_robots_blue,
                n_robots_yellow=args.env_n_robots_yellow,
                hierarchical=False
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
    val_env.set_key(key)

    m_observation_space, m_action_space = env.get_spaces_m()
    w_observation_space, w_action_space = env.get_spaces_w()

    agent = DDPG(m_observation_space, w_action_space, learning_rate, gamma, seed)
    buffer = ReplayBuffer(m_observation_space, w_action_space, replay_capacity)

    obs = env.reset()
    for step in tqdm(range(total_training_steps), smoothing=0.01):
        if args.training_val_frequency and step % args.training_val_frequency == 0:
            run_validation_ep(agent, val_env, opponent_policies)

        action = np.array(agent.sample_action(obs))
        _obs, reward, done, info = env.step(action)
        terminal_state = False if not done or "TimeLimit.truncated" in info else True
        buffer.add(obs, action, 0.0, reward.manager, terminal_state, _obs.manager)

        if step > min_replay_size:
            batch = buffer.get_batch(batch_size)
            agent.update(batch)

        obs = _obs.manager
        if done:
            obs = env.reset()


if __name__ == '__main__':
    parser = ArgumentParser(fromfile_prefix_chars='@')
    # RANDOM
    parser.add_argument('--seed', type=int, default=0)

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
    parser.add_argument('--training-total-steps', type=int, default=3000000)
    parser.add_argument('--training-replay-min-size', type=int, default=100000)
    parser.add_argument('--training-batch-size', type=int, default=256)
    parser.add_argument('--training-gamma', type=float, default=0.95)
    parser.add_argument('--training-learning-rate', type=float, default=1e-4)
    parser.add_argument('--training-val-frequency', type=int, default=100000)
    parser.add_argument('--training-load-worker', type=bool, default=True)

    args = parser.parse_args()
    main(args)
