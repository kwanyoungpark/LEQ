"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import gym
import os
import torch
import flax.linen as nn

import policy
import value_net
from algos.leq.actor import (
    update_actor_bc,
    DPG_lambda_update_actor,
    DPG_multistep_update_actor,
    onestep_update_actor,
)
from common import Batch, InfoDict, Model, PRNGKey, Params, get_deter
from algos.leq.critic import (
    update_q_fqe,
    onestep_update_q,
    multistep_update_q,
    lambda_update_q,
)

from functools import partial


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


@partial(jax.jit, static_argnames=["rollout_length"])
def _rollout(
    key: PRNGKey,
    observations: jnp.ndarray,
    rollout_length: int,
    actor: Model,
    model: Model,
    temperature: float = 1.0,
) -> np.ndarray:

    states, actions, rewards, masks = [observations], [], [], []

    for _ in range(rollout_length):
        key, rng1, rng2 = jax.random.split(key, 3)
        _, action = policy.sample_actions(rng1, actor, states[-1], temperature)
        next_obs, reward, terminal, _ = model(rng2, states[-1], action)
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


@jax.jit
def _update_bc_jit(actor: Model, data_batch: Batch) -> Tuple[Model, InfoDict]:

    new_actor, actor_info = update_actor_bc(actor, data_batch)

    return new_actor, {
        **actor_info,
    }


@jax.jit
def _update_fqe_jit(
    critic: Model,
    target_critic: Model,
    actor: Model,
    data_batch: Batch,
    discount: float,
    soft_target_update: float,
) -> Tuple[Model, InfoDict]:

    new_critic, critic_info = update_q_fqe(
        critic, target_critic, actor, data_batch, discount
    )
    new_target_critic = target_update(new_critic, target_critic, soft_target_update)

    return (
        new_critic,
        new_target_critic,
        {
            **critic_info,
        },
    )


@partial(
    jax.jit,
    static_argnames=["horizon_length", "num_repeat", "actor_update", "critic_update"],
)
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    model: Model,
    data_batch: Batch,
    model_batch: Batch,
    model_batch_ratio: float,
    discount: float,
    soft_target_update: float,
    expectile: float,
    lamb: float,
    horizon_length: int,
    num_repeat: int,
    actor_update: str,
    critic_update: str,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, InfoDict]:

    key, key2, key3, rng = jax.random.split(rng, 4)

    ## Update actor
    if actor_update == "lambda-return":
        # Directly optimize lower-expectile of lambda-returns
        new_actor, actor_info = DPG_lambda_update_actor(
            key,
            actor,
            critic,
            model,
            model_batch,
            discount,
            lamb,
            horizon_length,
            expectile,
            num_repeat,
        )
    elif actor_update == "multi-step":
        # Directly optimize lower-expectile of multi-step returns
        new_actor, actor_info = DPG_multistep_update_actor(
            key,
            actor,
            critic,
            model,
            model_batch,
            discount,
            horizon_length,
            expectile,
            num_repeat,
        )
    elif actor_update == "one-step":
        # Optimize Q(s,a)
        new_actor, actor_info = onestep_update_actor(
            key,
            actor,
            critic,
            model,
            model_batch,
            discount,
            horizon_length,
            expectile,
            num_repeat,
        )
    else:
        assert False, f"actor_update: {actor_update}"

    ## Update critic
    if critic_update == "lambda-return":
        # Use lower-expectile of lambda-returns as target value
        new_critic, critic_info = lambda_update_q(
            key3,
            critic,
            target_critic,
            actor,
            model,
            data_batch,
            model_batch,
            model_batch_ratio,
            discount,
            lamb,
            horizon_length,
            expectile,
            num_repeat,
        )
    elif critic_update == "multi-step":
        # Use lower-expectile of multi-step returns as target value
        new_critic, critic_info = multistep_update_q(
            key3,
            critic,
            target_critic,
            actor,
            model,
            data_batch,
            model_batch,
            model_batch_ratio,
            discount,
            horizon_length,
            expectile,
            num_repeat,
        )
    elif critic_update == "one-step":
        # Use lower-expectile of 1-step returns as target value
        new_critic, critic_info = onestep_update_q(
            key3,
            critic,
            target_critic,
            actor,
            model,
            data_batch,
            model_batch,
            model_batch_ratio,
            discount,
            horizon_length,
            expectile,
            num_repeat,
        )
    else:
        assert False, f"critic_update: {critic_update}"

    new_target_critic = target_update(new_critic, target_critic, soft_target_update)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        {
            **critic_info,
            **actor_info,
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
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        soft_target_update: float = 0.005,
        expectile: float = 0.1,
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
        num_repeat: int = None,
        actor_update: str = None,
        critic_update: str = None,
        **kwargs,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        self.expectile = expectile
        self.soft_target_update = soft_target_update
        self.discount = discount
        self.horizon_length = horizon_length
        self.lamb = lamb
        self.num_repeat = num_repeat
        self.actor_update = actor_update
        self.critic_update = critic_update

        assert actor_update in ["lambda-return", "multi-step", "one-step"], actor_update
        assert critic_update in [
            "lambda-return",
            "multi-step",
            "one-step",
        ], critic_update

        scaler = jnp.concatenate(jax.device_put(scaler), axis=0)  # [2, D]
        reward_scaler = jnp.stack(jax.device_put(reward_scaler), axis=0)  # [2,]
        obs_scaler = jax.device_put((jax.device_get(scaler)[:, :obs_dim]))
        # print(obs_scaler.shape, scaler.shape)

        rng = PRNGKey(seed)
        rng, model_key, actor_key, critic_key, value_key = jax.random.split(rng, 5)

        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            obs_dim,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=True,
            use_norm=False,
        )

        if opt_decay_schedule == "cosine":
            # schedule_fn = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=-actor_lr,
            #                                                 warmup_steps=int(max_steps*0.2),
            #                                                 decay_steps=max_steps, end_value=0.)
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

        # critic_def = value_net.DoubleCritic(scaler, hidden_dims, use_norm=True)
        critic_def = value_net.Critic(scaler, hidden_dims, use_norm=True)
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
                key, observations, rollout_length, self.actor, self.model, temperature
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

    def update_bc(self, data_batch: Batch):
        data_batch = self.preprocess(data_batch)
        new_actor_pretrain, info = _update_bc_jit(self.actor_pretrain, data_batch)
        self.actor_pretrain = new_actor_pretrain
        return info

    def update_fqe(self, data_batch: Batch):
        data_batch = self.preprocess(data_batch)
        new_critic_pretrain, new_target_critic_pretrain, info = _update_fqe_jit(
            self.critic_pretrain,
            self.target_critic_pretrain,
            self.actor,
            data_batch,
            self.discount,
            self.soft_target_update,
        )
        self.critic_pretrain = new_critic_pretrain
        self.target_critic_pretrain = new_target_critic_pretrain
        return info

    def update(
        self, data_batch: Batch, model_batch: Batch, model_batch_ratio: float
    ) -> InfoDict:
        data_batch, model_batch = self.preprocess(data_batch), self.preprocess(
            model_batch
        )
        new_rng, new_actor, new_critic, new_target_critic, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.model,
            data_batch,
            model_batch,
            model_batch_ratio,
            self.discount,
            self.soft_target_update,
            self.expectile,
            self.lamb,
            self.horizon_length,
            self.num_repeat,
            self.actor_update,
            self.critic_update,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info
