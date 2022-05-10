import time
from argparse import ArgumentParser
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import gym
import jax
import numpy as np
import pyvirtualdisplay
import rsoccer_gym
import rsoccer_gym.experimental
import wandb
from flax.training import checkpoints
from tqdm import tqdm

from agent import DDPG
from buffer import ReplayBuffer


def info_to_log(info):
    return {
        'manager/goal': info['manager_weighted_rw'][0],
        'manager/ball_grad': info['manager_weighted_rw'][1],
        'manager/move': info['manager_weighted_rw'][2],
        'manager/energy': info['manager_weighted_rw'][4],
    }


def run_validation_ep(agent, env, opponent_policies):
    obs = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        action = np.array(action.reshape((-1, 2)))
        step_action = np.concatenate(
            [action] + [[p()] for p in opponent_policies], axis=0
        )
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
    total_training_steps = args.training_total_steps
    replay_capacity = args.training_total_steps
    min_replay_size = args.training_replay_min_size
    batch_size = args.training_batch_size
    gamma_m = args.training_gamma_manager
    sigma_m = args.training_noise_sigma_manager
    learning_rate_actor = args.training_learning_rate_actor
    learning_rate_critic = args.training_learning_rate_critic
    seed = args.seed
    n_controlled_robots = args.env_n_robots_blue
    nsteps_per_grad = args.training_nsteps_per_grad
    ngrad_per_update = args.training_ngrads_per_update
    val_frequency = args.training_val_frequency

    env = gym.make(
        args.env_name,
        n_robots_blue=args.env_n_robots_blue,
        n_robots_yellow=args.env_n_robots_yellow,
        hierarchical=False,
    )
    if args.training_val_frequency:
        val_env = gym.wrappers.RecordVideo(
            gym.make(
                args.env_name,
                n_robots_blue=args.env_n_robots_blue,
                n_robots_yellow=args.env_n_robots_yellow,
                hierarchical=False,
            ),
            f'./monitor/{args.wandb_name}/',
            episode_trigger=lambda x: True,
        )
    key = jax.random.PRNGKey(seed)
    if args.env_opponent_policy == 'off':
        opponent_policies = [lambda: np.array([0.0, 0.0]) for _ in range(args.env_n_robots_yellow)]
    env.set_key(key)
    val_env.set_key(key) if args.training_val_frequency else None

    agent = DDPG(
        env.observation_space,
        env.action_space,
        learning_rate_actor,
        learning_rate_critic,
        gamma_m,
        seed,
        sigma_m,
    )

    # Non hiearquical needs to cotrol all blue robots, 
    # w_action space is for only one robot and env action space is for blue and yellow robots
    action_space = gym.spaces.Box(
        low=-1,
        high=1,
        shape=((args.env_n_robots_blue) * 2,),
        dtype=np.float32,
    )
    buffer = ReplayBuffer(env.observation_space, action_space, replay_capacity)

    @jax.jit
    def random_action(k):
        k1, k2 = jax.random.split(k, 2)
        return k1, jax.random.uniform(k2, shape=(n_controlled_robots*2,))

    obs = env.reset()
    rewards, ep_steps, n_grads, done, q_losses, pi_losses = 0, 0, 0, False, [], []
    for step in tqdm(range(total_training_steps), smoothing=0.01):
        buffering = step < min_replay_size

        if val_frequency and not buffering and step % val_frequency == 0:
            run_validation_ep(agent, val_env, opponent_policies)

        if buffering:
            key, action = random_action(key)
            action = np.array(action)
        else:
            action = np.array(agent.sample_action(obs))

        step_action = np.concatenate(
            [action.reshape((-1, 2))] + [[p()] for p in opponent_policies], axis=0
        )
        _obs, reward, done, info = env.step(step_action)
        terminal_state = False if not done or "TimeLimit.truncated" in info else True
        buffer.add(obs, action, reward.manager, terminal_state, _obs.manager)

        rewards += reward.manager
        ep_steps += 1
        if step >= min_replay_size:
            if step % nsteps_per_grad == 0:
                for i in range(ngrad_per_update):
                    batch = buffer.get_batch(batch_size)
                    act_loss, crt_loss = agent.update(batch)
                    pi_losses.append(act_loss)
                    q_losses.append(crt_loss)
                    n_grads += 1

        obs = _obs.manager
        if done:
            log = info_to_log(info)
            log.update(dict(ep_reward=rewards, ep_steps=ep_steps))
            if len(q_losses):
                log.update(
                    q_loss=sum(q_losses) / len(q_losses),
                    pi_loss=sum(pi_losses) / len(pi_losses),
                    n_grads=n_grads,
                )
            wandb.log(log, step=step)
            obs = env.reset()
            rewards, ep_steps, q_losses, pi_losses = 0, 0, [], []

    checkpoints.save_checkpoint(
        f'./checkpoints/{args.env_name}/{args.wandb_name}',
        agent.actor_params,
        step=total_training_steps,
        overwrite=True,
    )


if __name__ == '__main__':
    # Creates a virtual display for OpenAI gym
    pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    parser = ArgumentParser(fromfile_prefix_chars='@')

    # EXPERIMENT
    parser.add_argument('--experiment-type', type=str, default='not-set')

    # RANDOM
    parser.add_argument('--seed', type=int, default=0)

    # WANDB
    parser.add_argument('--wandb-mode', type=str, default='disabled')
    parser.add_argument('--wandb-project', type=str, default='rsoccer-hrl')
    parser.add_argument('--wandb-entity', type=str, default='felipemartins')
    parser.add_argument('--wandb-name', type=str)
    parser.add_argument('--wandb-monitor-gym', type=bool, default=True)

    # ENVIRONMENT
    parser.add_argument('--env-name', type=str, default='VSSHRL-v5')
    parser.add_argument('--env-n-robots-blue', type=int, default=1)
    parser.add_argument('--env-n-robots-yellow', type=int, default=0)
    parser.add_argument('--env-opponent-policy', type=str, default='off')

    # TRAINING
    parser.add_argument('--training-total-steps', type=int, default=20000000)
    parser.add_argument('--training-replay-min-size', type=int, default=100000)
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--training-gamma-manager', type=float, default=0.95)
    parser.add_argument('--training-gamma-worker', type=float, default=0.95)
    parser.add_argument('--training-learning-rate-actor', type=float, default=1e-4)
    parser.add_argument('--training-learning-rate-critic', type=float, default=2e-4)
    parser.add_argument('--training-val-frequency', type=int, default=250000)
    parser.add_argument('--training-load-worker', type=bool, default=True)
    parser.add_argument('--training-nsteps-per-grad', type=int, default=10)
    parser.add_argument('--training-ngrads-per-update', type=int, default=1)
    parser.add_argument('--training-noise-sigma-manager', type=float, default=0.2)
    parser.add_argument('--training-noise-sigma-worker', type=float, default=0.2)

    args = parser.parse_args()
    main(args)
