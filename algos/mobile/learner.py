"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import gym
import torch
import flax.linen as nn

import policy
import value_net
from algos.mobile.actor import update_actor, update_alpha
from common import Batch, InfoDict, Model, PRNGKey, Params
from algos.mobile.critic import update_q

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


def _replace(model: Model, params: Params) -> Model:
    new_params = model.params
    for k, v in params.items():
        new_params[k] = v
    return model.replace(params=new_params)


def run_model(key, model, states, actions):
    key1, key2 = jax.random.split(key, 2)
    s_next, rew, terminals, _ = model(key1, states, actions)
    model_idxs = jax.random.choice(key2, model.apply_fn.elites, states.shape[:-1])
    s_next = jnp.take_along_axis(s_next, model_idxs[None, :, None], axis=0)[0]
    rew = jnp.take_along_axis(rew, model_idxs[None, :], axis=0)[0]
    terminals = jnp.take_along_axis(terminals, model_idxs[None, :], axis=0)[0]
    return s_next, rew, terminals, None


@partial(jax.jit, static_argnames=["rollout_length"])
def _rollout(
    key: PRNGKey,
    observations: jnp.ndarray,
    rollout_length: int,
    actor: Model,
    model: Model,
    temperature: float = 1.0,
) -> np.ndarray:

    batch_size = observations.shape[0]
    states, actions, rewards, masks = [observations], [], [], []
    for _ in range(rollout_length):
        key, rng1, rng2 = jax.random.split(key, 3)
        _, action = policy.sample_actions(rng1, actor, states[-1], temperature)
        next_obs, reward, terminal, _ = run_model(rng2, model, states[-1], action)
        states.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        masks.append(1 - terminal)
    obss = jnp.concatenate(states[:-1], axis=0)
    next_obss = jnp.concatenate(states[1:], axis=0)
    actions = jnp.concatenate(actions, axis=0)
    rewards = jnp.concatenate(rewards, axis=0)
    masks = jnp.concatenate(masks, axis=0)
    return {
        "obss": obss,
        "actions": actions,
        "rewards": rewards,
        "masks": masks,
        "next_obss": next_obss,
    }


@partial(
    jax.jit,
    static_argnames=[
        "horizon_length",
        "num_repeat",
    ],
)
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    sac_alpha: Model,
    critic: Model,
    target_critic: Model,
    model: Model,
    data_batch: Batch,
    model_batch: Batch,
    model_batch_ratio: float,
    discount: float,
    tau: float,
    beta: float,
    temperature: float,
    target_entropy: float,
    lamb: float,
    horizon_length: int,
    num_repeat: int,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, InfoDict]:

    log_alpha = sac_alpha()
    alpha = jnp.exp(log_alpha)

    key, key2, key3, rng = jax.random.split(rng, 4)

    mix_batch = Batch(
        observations=jnp.concatenate(
            [data_batch.observations, model_batch.observations], axis=0
        ),
        actions=jnp.concatenate([data_batch.actions, model_batch.actions], axis=0),
        rewards=jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0),
        masks=jnp.concatenate([data_batch.masks, model_batch.masks], axis=0),
        next_observations=jnp.concatenate(
            [data_batch.next_observations, model_batch.next_observations], axis=0
        ),
        returns_to_go=None,
    )
    new_critic, critic_info = update_q(
        key3,
        critic,
        target_critic,
        actor,
        model,
        data_batch,
        model_batch,
        discount,
        temperature,
        lamb,
        beta,
        num_repeat,
    )
    new_actor, actor_info = update_actor(
        key, actor, critic, mix_batch, discount, temperature, alpha,
    )
    
    new_alpha, alpha_info = update_alpha(
        key2, actor_info["log_probs"], sac_alpha, target_entropy
    )

    new_target_critic = target_update(new_critic, target_critic, tau)

    return (
        rng,
        new_actor,
        new_alpha,
        new_critic,
        new_target_critic,
        {
            **critic_info,
            **actor_info,
            **alpha_info,
        },
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        dynamics_name: str = None,
        env_name: str = None,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        beta: float = None,
        temperature: float = 1.0,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        model: Model = None,
        opt_decay_schedule: str = "cosine",
        num_models: int = 7,
        num_elites: int = 5,
        model_hidden_dims: Sequence[int] = (256, 256, 256),
        lamb: float = 0.95,
        horizon_length: int = None,
        scaler: Tuple[np.ndarray, np.ndarray] = None,
        reward_scaler: Tuple[float, float] = None,
        num_actor_updates: int = None,
        baseline: str = None,
        num_repeat: int = None,
        # sac_alpha: float = 0.2,
        **kwargs
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        self.beta = beta
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.target_entropy = -action_dim
        self.horizon_length = horizon_length
        self.lamb = lamb
        self.num_actor_updates = num_actor_updates
        self.num_repeat = num_repeat

        scaler = jnp.concatenate(jax.device_put(scaler), axis=0)  # [2, D]
        reward_scaler = jnp.stack(jax.device_put(reward_scaler), axis=0)  # [2,]
        obs_scaler = jax.device_put((jax.device_get(scaler)[:, :obs_dim]))

        rng = PRNGKey(seed)
        rng, model_key, actor_key, alpha_key, critic_key, value_key = jax.random.split(
            rng, 6
        )

        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            obs_dim,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=True,
            use_norm=False,
            use_symlog=True,
        )

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_optimiser = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            actor_optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(
            actor_def, inputs=[actor_key, observations], tx=actor_optimiser
        )
        actor = _replace(actor, {"scaler": obs_scaler})

        alpha_def = policy.SACalpha()
        alpha = Model.create(
            alpha_def, inputs=[alpha_key], tx=optax.adam(learning_rate=alpha_lr)
        )

        critic_def = value_net.Critic(
            scaler, hidden_dims, use_norm=True, use_symlog=True
        )
        critic_opt = optax.adam(learning_rate=value_lr)
        critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions], tx=critic_opt
        )

        value_def = value_net.ValueCritic(obs_scaler, hidden_dims)
        value_opt = optax.adam(learning_rate=value_lr)
        value = Model.create(value_def, inputs=[value_key, observations], tx=value_opt)

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )
        target_value = Model.create(value_def, inputs=[value_key, observations])

        actor_pretrain = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=optax.adam(learning_rate=actor_lr),
        )
        actor_pretrain = _replace(actor_pretrain, {"scaler": obs_scaler})
        critic_pretrain = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=value_lr),
        )
        target_critic_pretrain = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=value_lr),
        )
        self.actor_pretrain = actor_pretrain
        self.critic_pretrain = critic_pretrain
        self.target_critic_pretrain = target_critic_pretrain

        self.actor = actor
        self.alpha = alpha
        self.critic = critic
        self.value = value
        self.model = model
        self.target_critic = target_critic
        self.target_value = target_value
        self.rng = rng

    def sample_actions(
        self, key: PRNGKey, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        observations = jax.device_put(observations)
        _, actions = policy.sample_actions(key, self.actor, observations, temperature)
        actions = jax.device_get(actions)
        return np.clip(actions, -1, 1)

    def rollout(
        self,
        key: PRNGKey,
        observations: np.ndarray,
        rollout_length: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        observations = jax.device_put(observations)
        with jax.transfer_guard("allow"):
            results = _rollout(
                key,
                observations,
                rollout_length,
                self.actor,
                self.model,
                temperature,
            )
        results = {k: jax.device_get(v) for (k, v) in results.items()}
        return results

    def preprocess(self, batch):
        new_batch = Batch(
            observations=jax.device_put(batch.observations),
            actions=jax.device_put(batch.actions),
            rewards=jax.device_put(batch.rewards),
            masks=jax.device_put(batch.masks),
            next_observations=jax.device_put(batch.next_observations),
            returns_to_go=None,
        )
        return new_batch

    def update(
        self, data_batch: Batch, model_batch: Batch, model_batch_ratio: float
    ) -> InfoDict:
        data_batch, model_batch = self.preprocess(data_batch), self.preprocess(
            model_batch
        )
        new_rng, new_actor, new_alpha, new_critic, new_target_critic, info = (
            _update_jit(
                self.rng,
                self.actor,
                self.alpha,
                self.critic,
                self.target_critic,
                self.model,
                data_batch,
                model_batch,
                model_batch_ratio,
                self.discount,
                self.tau,
                self.beta,
                self.temperature,
                self.target_entropy,
                self.lamb,
                self.horizon_length,
                self.num_repeat,
            )
        )

        self.rng = new_rng
        self.actor = new_actor
        self.alpha = new_alpha
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info
