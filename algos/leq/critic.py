from typing import Tuple

import jax.numpy as jnp
import jax

import numpy as np
from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss, get_deter


def update_q_fqe(
    critic: Model, target_critic: Model, actor: Model, batch: Batch, discount: float
):
    next_a = get_deter(actor(batch.next_observations))
    next_value = critic(batch.next_observations, next_a)
    target_q_data = batch.rewards + batch.masks * discount * next_value

    target_q = target_critic(batch.next_observations, next_a)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = jnp.mean((q - target_q_data) ** 2)
        critic_reg_loss = jnp.mean((q - target_q) ** 2)
        return critic_loss + critic_reg_loss, {
            "critic_loss": critic_loss,
            "critic_reg_loss": critic_reg_loss,
            "reward": batch.rewards.mean(),
            "reward_min": batch.rewards.min(),
            "reward_max": batch.rewards.max(),
            "q": q.mean(),
            "q_min": q.min(),
            "q_max": q.max(),
            "target_q_data": target_q_data.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info


def onestep_update_q(
    key: PRNGKey,
    critic: Model,
    target_critic: Model,
    actor: Model,
    model: Model,
    data_batch: Batch,
    model_batch: Batch,
    model_batch_ratio: float,
    discount: float,
    H: int,
    expectile: float,
    num_repeat: int,
) -> Tuple[Model, Model, InfoDict]:

    N = model_batch.observations.shape[0]
    key1, key2, key3, key4 = jax.random.split(key, 4)

    ## Generate imaginary trajectories
    Robs = (
        model_batch.observations[:, None, :]
        .repeat(repeats=num_repeat, axis=1)
        .reshape(N * num_repeat, -1)
    )
    Ra = get_deter(actor(Robs, 0.0))
    states, rewards, actions, masks, mask_weights, loss_weights = (
        [Robs],
        [],
        [Ra],
        [],
        [jnp.ones(N * num_repeat)],
        [jnp.ones(N * num_repeat)],
    )
    for i in range(H):
        s, a = states[-1], actions[-1]
        rng1, rng2, key1 = jax.random.split(key1, 3)
        s_next, rew, terminal, _ = model(rng2, s, jnp.clip(a, -1.0, 1.0))
        a_next = get_deter(actor(s_next, 0.0))
        states.append(s_next)
        actions.append(a_next)
        rewards.append(rew)
        masks.append(1 - terminal)
        mask_weights.append(mask_weights[i] * (1 - terminal))
        loss_weights.append(loss_weights[i] * (1 - terminal) * discount)

    mask_weights = jnp.stack(mask_weights, axis=0)
    loss_weights = jnp.stack(loss_weights[:-1], axis=0)

    ## Calculate one-step returns
    target_q_rollout = []
    for i in range(H):
        target_q_rollout.append(
            rewards[i] + masks[i] * discount * critic(states[i + 1], actions[i + 1])
        )

    target_q_rollout = jnp.stack(target_q_rollout, axis=0)
    states = jnp.stack(states[:-1], axis=0)
    actions = jnp.stack(actions[:-1], axis=0)
    rewards = jnp.stack(rewards, axis=0)

    ## Calculate target for dataset transitions
    next_a = get_deter(actor(data_batch.next_observations, 0.0))
    next_value = critic(data_batch.next_observations, next_a)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_value

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Calculate critic loss for dataset transitions
        q_data = critic.apply(
            {"params": critic_params}, data_batch.observations, data_batch.actions
        )
        critic_loss_data = loss(target_q_data, q_data, 0.5).mean()

        # Calculate critic loss for imaginated trajectories
        q_rollout = critic.apply({"params": critic_params}, states, actions)
        critic_loss_rollout = loss(target_q_rollout, q_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).mean()

        # EMA regularization loss
        q_target_data = target_critic(data_batch.observations, data_batch.actions)
        critic_reg_loss_data = jnp.mean((q_target_data - q_data) ** 2)
        q_target_rollout = target_critic(states, actions)
        critic_reg_loss_rollout = (q_target_rollout - q_rollout) ** 2
        critic_reg_loss_rollout = jnp.mean(critic_reg_loss_rollout * loss_weights)

        critic_reg_loss = (
            critic_reg_loss_rollout * model_batch_ratio
            + critic_reg_loss_data * (1 - model_batch_ratio)
        )
        return critic_loss_data + critic_loss_rollout + critic_reg_loss, {
            "critic_loss_data": critic_loss_data.mean(),
            "critic_loss_model": critic_loss_rollout.mean(),
            "q_data": q_data.mean(),
            "q_data_min": q_data.min(),
            "q_data_max": q_data.max(),
            "reward_data": data_batch.rewards.mean(),
            "reward_data_max": data_batch.rewards.max(),
            "reward_data_min": data_batch.rewards.min(),
            "q_model": q_rollout.mean(),
            "q_model_min": q_rollout.min(),
            "q_model_max": q_rollout.max(),
            "reward_model": (rewards * mask_weights[:-1]).mean(),
            "reward_max": (rewards * mask_weights[:-1]).max(),
            "reward_min": (rewards * mask_weights[:-1]).min(),
            "critic_reg_loss": critic_reg_loss,
            "critic_reg_loss_data": critic_reg_loss_data,
            "critic_reg_loss_rollout": critic_reg_loss_rollout,
        }

    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)

    infos = {
        "mask_weights": mask_weights.mean(),
        "loss_weights": loss_weights.mean(),
        "expectile": expectile,
        "model_batch_ratio": model_batch_ratio,
        "state_max": jnp.abs(states * mask_weights[:-1, :, None]).max(),
    }

    return new_critic, {
        **infos,
        **critic_info,
    }


def multistep_update_q(
    key: PRNGKey,
    critic: Model,
    target_critic: Model,
    actor: Model,
    model: Model,
    data_batch: Batch,
    model_batch: Batch,
    model_batch_ratio: float,
    discount: float,
    H: int,
    expectile: float,
    num_repeat: int,
) -> Tuple[Model, Model, InfoDict]:

    N = model_batch.observations.shape[0]
    key1, key2, key3, key4 = jax.random.split(key, 4)

    ## Generate imaginary trajectories
    Robs = (
        model_batch.observations[:, None, :]
        .repeat(repeats=num_repeat, axis=1)
        .reshape(N * num_repeat, -1)
    )
    Ra = get_deter(actor(Robs, 0.0))
    states, rewards, actions, masks, mask_weights, loss_weights = (
        [Robs],
        [],
        [Ra],
        [],
        [jnp.ones(N * num_repeat)],
        [jnp.ones(N * num_repeat)],
    )
    for i in range(H):
        s, a = states[-1], actions[-1]
        rng1, rng2, key1 = jax.random.split(key1, 3)
        s_next, rew, terminal, _ = model(rng2, s, jnp.clip(a, -1.0, 1.0))
        a_next = get_deter(actor(s_next, 0.0))
        states.append(s_next)
        actions.append(a_next)
        rewards.append(rew)
        masks.append(1 - terminal)
        mask_weights.append(mask_weights[i] * (1 - terminal))
        loss_weights.append(loss_weights[i] * (1 - terminal) * discount)

    mask_weights = jnp.stack(mask_weights, axis=0)
    loss_weights = jnp.stack(loss_weights[:-1], axis=0)

    ## Calculate multi-step returns
    target_q_rollout = [critic(states[-1], actions[-1])]
    for i in reversed(range(H)):
        q_cur = critic(states[i], actions[i])
        q_next = (
            mask_weights[i] * rewards[i]
            + mask_weights[i + 1] * discount * target_q_rollout[-1]
        )
        target_q_rollout.append(q_next)
    target_q_rollout = list(reversed(target_q_rollout))[:-1]

    target_q_rollout = jnp.stack(target_q_rollout, axis=0)
    states = jnp.stack(states[:-1], axis=0)
    actions = jnp.stack(actions[:-1], axis=0)
    rewards = jnp.stack(rewards, axis=0)

    ## Calculate target for dataset transitions
    next_a = get_deter(actor(data_batch.next_observations, 0.0))
    next_value = critic(data_batch.next_observations, next_a)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_value

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Calculate critic loss for dataset transitions
        q_data = critic.apply(
            {"params": critic_params}, data_batch.observations, data_batch.actions
        )
        critic_loss_data = loss(target_q_data, q_data, 0.5).mean()

        # Calculate critic loss for imaginated trajectories
        q_rollout = critic.apply({"params": critic_params}, states, actions)
        critic_loss_rollout = loss(target_q_rollout, q_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).mean()

        # EMA regularization loss
        q_target_data = target_critic(data_batch.observations, data_batch.actions)
        critic_reg_loss_data = jnp.mean((q_target_data - q_data) ** 2)
        q_target_rollout = target_critic(states, actions)
        critic_reg_loss_rollout = (q_target_rollout - q_rollout) ** 2
        critic_reg_loss_rollout = jnp.mean(critic_reg_loss_rollout * loss_weights)

        critic_reg_loss = (
            critic_reg_loss_rollout * model_batch_ratio
            + critic_reg_loss_data * (1 - model_batch_ratio)
        )
        return critic_loss_data + critic_loss_rollout + critic_reg_loss, {
            "critic_loss_data": critic_loss_data.mean(),
            "critic_loss_model": critic_loss_rollout.mean(),
            "q_data": q_data.mean(),
            "q_data_min": q_data.min(),
            "q_data_max": q_data.max(),
            "reward_data": data_batch.rewards.mean(),
            "reward_data_max": data_batch.rewards.max(),
            "reward_data_min": data_batch.rewards.min(),
            "q_model": q_rollout.mean(),
            "q_model_min": q_rollout.min(),
            "q_model_max": q_rollout.max(),
            "reward_model": (rewards * mask_weights[:-1]).mean(),
            "reward_max": (rewards * mask_weights[:-1]).max(),
            "reward_min": (rewards * mask_weights[:-1]).min(),
            "critic_reg_loss": critic_reg_loss,
            "critic_reg_loss_data": critic_reg_loss_data,
            "critic_reg_loss_rollout": critic_reg_loss_rollout,
        }

    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)

    infos = {
        "mask_weights": mask_weights.mean(),
        "loss_weights": loss_weights.mean(),
        "expectile": expectile,
        "model_batch_ratio": model_batch_ratio,
        "state_max": jnp.abs(states * mask_weights[:-1, :, None]).max(),
    }

    return new_critic, {
        **infos,
        **critic_info,
    }


def lambda_update_q(
    key: PRNGKey,
    critic: Model,
    target_critic: Model,
    actor: Model,
    model: Model,
    data_batch: Batch,
    model_batch: Batch,
    model_batch_ratio: float,
    discount: float,
    lamb: float,
    H: int,
    expectile: float,
    num_repeat: int,
) -> Tuple[Model, Model, InfoDict]:

    N = model_batch.observations.shape[0]
    key1, key2, key3, key4 = jax.random.split(key, 4)

    ## Generate imaginary trajectories
    Robs = (
        model_batch.observations[:, None, :]
        .repeat(repeats=num_repeat, axis=1)
        .reshape(N * num_repeat, -1)
    )
    Ra = get_deter(actor(Robs, 0.0))
    states, rewards, actions, masks, mask_weights, loss_weights = (
        [Robs],
        [],
        [Ra],
        [],
        [jnp.ones(N * num_repeat)],
        [jnp.ones(N * num_repeat)],
    )
    for i in range(H):
        s, a = states[-1], actions[-1]
        rng1, rng2, key1 = jax.random.split(key1, 3)
        s_next, rew, terminal, _ = model(rng2, s, jnp.clip(a, -1.0, 1.0))
        a_next = get_deter(actor(s_next, 0.0))
        states.append(s_next)
        actions.append(a_next)
        rewards.append(rew)
        masks.append(1 - terminal)
        mask_weights.append(mask_weights[i] * (1 - terminal))
        loss_weights.append(loss_weights[i] * (1 - terminal) * discount)

    mask_weights = jnp.stack(mask_weights, axis=0)
    loss_weights = jnp.stack(loss_weights[:-1], axis=0)

    ## Calculate lambda-returns
    target_q_rollout, lamb_weight = [critic(states[-1], actions[-1])], 1.0
    for i in reversed(range(H)):
        q_cur = critic(states[i], actions[i])
        q_next = (
            mask_weights[i] * rewards[i]
            + mask_weights[i + 1] * discount * target_q_rollout[-1]
        )
        next_value = (q_cur + lamb * lamb_weight * q_next) / (1 + lamb * lamb_weight)
        target_q_rollout.append(next_value)
        lamb_weight = 1.0 + lamb * lamb_weight
    target_q_rollout = list(reversed(target_q_rollout))[:-1]

    target_q_rollout = jnp.stack(target_q_rollout, axis=0)
    states = jnp.stack(states[:-1], axis=0)
    actions = jnp.stack(actions[:-1], axis=0)
    rewards = jnp.stack(rewards, axis=0)

    ## Calculate target for dataset transitions
    next_a = get_deter(actor(data_batch.next_observations, 0.0))
    next_value = critic(data_batch.next_observations, next_a)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_value

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Calculate critic loss for dataset transitions
        q_data = critic.apply(
            {"params": critic_params}, data_batch.observations, data_batch.actions
        )
        critic_loss_data = loss(target_q_data, q_data, 0.5).mean()

        # Calculate critic loss for imaginated trajectories
        q_rollout = critic.apply({"params": critic_params}, states, actions)
        critic_loss_rollout = loss(target_q_rollout, q_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).mean()

        # EMA regularization loss
        q_target_data = target_critic(data_batch.observations, data_batch.actions)
        critic_reg_loss_data = jnp.mean((q_target_data - q_data) ** 2)
        q_target_rollout = target_critic(states, actions)
        critic_reg_loss_rollout = (q_target_rollout - q_rollout) ** 2
        critic_reg_loss_rollout = jnp.mean(critic_reg_loss_rollout * loss_weights)

        critic_reg_loss = (
            critic_reg_loss_rollout * model_batch_ratio
            + critic_reg_loss_data * (1 - model_batch_ratio)
        )
        return critic_loss_data * (
            1 - model_batch_ratio
        ) + critic_loss_rollout * model_batch_ratio + critic_reg_loss, {
            "critic_loss_data": critic_loss_data.mean(),
            "critic_loss_model": critic_loss_rollout.mean(),
            "q_data": q_data.mean(),
            "q_data_min": q_data.min(),
            "q_data_max": q_data.max(),
            "reward_data": data_batch.rewards.mean(),
            "reward_data_max": data_batch.rewards.max(),
            "reward_data_min": data_batch.rewards.min(),
            "q_model": q_rollout.mean(),
            "q_model_min": q_rollout.min(),
            "q_model_max": q_rollout.max(),
            "reward_model": (rewards * mask_weights[:-1]).mean(),
            "reward_max": (rewards * mask_weights[:-1]).max(),
            "reward_min": (rewards * mask_weights[:-1]).min(),
            "critic_reg_loss": critic_reg_loss,
            "critic_reg_loss_data": critic_reg_loss_data,
            "critic_reg_loss_rollout": critic_reg_loss_rollout,
        }

    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)

    infos = {
        "mask_weights": mask_weights.mean(),
        "loss_weights": loss_weights.mean(),
        "expectile": expectile,
        "model_batch_ratio": model_batch_ratio,
        "state_max": jnp.abs(states * mask_weights[:-1, :, None]).max(),
    }

    return new_critic, {
        **infos,
        **critic_info,
    }
