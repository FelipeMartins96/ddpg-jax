import gym
from gym.wrappers import RecordVideo
import rsoccer_gym.experimental
from agent import DDPG
from buffer import ReplayBuffer
import wandb
import numpy as np
from tqdm import tqdm
from flax.training import checkpoints

ENVIRONMENT_STEPS= int(5e6)
MIN_BUFFER = 100000
N_ENVS = 5
VAL_FREQ = 10000
def run_validation_ep(agent, env):
    obs = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        _obs, _, done, _ = env.step(action)
        obs = _obs


wandb.init(
    project='msc-w23',
    entity='felipemartins',
    monitor_gym=True,
)

envs = gym.vector.make("msc-v230", num_envs=N_ENVS, asynchronous=True)
val_env = gym.make("msc-v230")
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

obs = envs.reset()
for step in tqdm(range(int(ENVIRONMENT_STEPS/N_ENVS)), smoothing=0.01):
    buffering = buffer.size < MIN_BUFFER

    # if VAL_FREQ and not buffering and step % VAL_FREQ == 0:
    #     run_validation_ep(agent, val_env)
    #     checkpoints.save_checkpoint(
    #         f'./checkpoints/',
    #         agent.actor_params,
    #         step=step,
    #         overwrite=True,
    #     )

    if not buffering:
        actions = np.array(agent.sample_action(obs))
    else:
        actions = envs.action_space.sample()

    n_obs, rewards, dones, infos = envs.step(actions)

    for i in range(N_ENVS):
        # terminal_state = False if not dones[i] or "TimeLimit.truncated" in infos[i] else True
        buffer.add(obs[i], actions[i], rewards[i], dones[i], n_obs[i])

    if buffer.size >= MIN_BUFFER:
        batch = buffer.get_batch(64)
        agent.update(batch)

    obs = n_obs
    if dones.any():
        log = {
        'goal': infos['rewards'][0][dones].mean(),
        'ball_grad': infos['rewards'][1][dones].mean(),
        'move': infos['rewards'][2][dones].mean(),
        'energy': infos['rewards'][4][dones].mean(),
        }
        wandb.log(log, step=step)

checkpoints.save_checkpoint(
    f'./checkpoints/',
    agent.actor_params,
    step=int(ENVIRONMENT_STEPS/N_ENVS),
    overwrite=True,
)