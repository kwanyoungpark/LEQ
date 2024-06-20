from typing import Dict, List

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle as pkl
import gym
import numpy as np
import copy
import time
from tqdm import tqdm
from functools import partial

from common import PRNGKey, Model


@jax.jit
def step_imagine(
    key: PRNGKey,
    model_eval: Model,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    states: jnp.ndarray,
):
    is_first = jnp.ones(obs.shape[0])
    next_states = model_eval(key, obs, action, is_first, states)
    return states


@partial(jax.jit, static_argnames=["action_dim", "state_dim"])
def step_imagine_first(
    key: PRNGKey, model_eval: Model, obs: jnp.ndarray, action_dim: int, state_dim: int
):
    batch_size = obs.shape[0]
    action = jnp.zeros((batch_size, action_dim))
    is_first = jnp.zeros(batch_size)
    states = jnp.zeros((batch_size, state_dim))
    next_states = model_eval(key, obs, action, is_first, states)
    return states


def evaluate(
    seed: int,
    agent: nn.Module,
    envs: List[gym.Env],
    video_path: str,
    step: int,
    model_eval=None,
    debug=False,
) -> Dict[str, float]:
    stats = {"return": [], "length": []}
    states, actions, rewards, observations = [], [], [], []

    _observations, dones = [], []
    num_episodes = len(envs)
    s = time.time()
    key = jax.device_put(PRNGKey(seed))
    for env in envs:
        _observations.append(env.reset())
        observations.append([])
        dones.append(False)
        states.append([])
        actions.append([])
        rewards.append([])
    # print(observations)
    _observations = np.array(_observations)

    dones = np.array(dones, dtype=bool)
    for j in tqdm(range(10000)):
        if np.all(dones):
            break
        if model_eval is not None:
            key, rng = jax.random.split(key)
            if j == 0:
                action_dim, state_dim = env.action_space.shape[-1], 32 * 32 + 200
                _states = step_imagine_first(
                    key, model_eval, _observations, action_dim, state_dim
                )
            else:
                _states = step_imagine(
                    key, model_eval, _observations, _actions, jax.device_put(_states)
                )
            _states = jax.device_get(_states)
        else:
            _states = _observations
        _actions = agent.sample_actions(key, _states, temperature=0.0)
        _actions = np.array(_actions)
        for i in range(len(envs)):
            if dones[i]:
                continue
            observations[i].append(np.copy(_observations[i]))
            states[i].append(np.copy(_states[i]))
            actions[i].append(np.copy(_actions[i]))
            obs, reward, done, info = envs[i].step(_actions[i])
            _observations[i] = obs
            rewards[i].append(reward)
            if done:
                dones[i] = True
                stats["return"].append(info["episode"]["return"])
                stats["length"].append(info["episode"]["length"])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    observations = np.concatenate(observations, axis=0)
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)

    if debug:
        _states, _actions = jax.device_put(states), jax.device_put(actions)
        q_values = agent.critic(_states, _actions)
        q_values = jax.device_get(q_values)
        trajectory = (observations, states, actions, rewards)
        print("Saving to:", video_path, step)
        np.save(os.path.join(video_path, f"q_values_{step}.npz"), q_values)
        with open(os.path.join(video_path, f"traj_{step}.pkl"), "wb") as F:
            pkl.dump(trajectory, F)
    return stats
