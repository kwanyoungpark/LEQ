from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss, get_deter, get_stoch

sg = lambda x: jax.lax.stop_gradient(x)


# MOBILE penalization (LCB)
def get_penalty(
    key: PRNGKey,
    critic: Model,
    actor: Model,
    model: Model,
    states: jnp.ndarray,
    actions: jnp.ndarray,
):
    N = states.shape[0]
    s_next, _, _, _ = model(key, states, actions)  # [7, N, num_repeat]
    a_next = get_deter(actor(s_next))
    q_next = critic(s_next, a_next)
    penalty = q_next.std(axis=0)
    return penalty


# Randomly select one model and use it
def run_model(key, model, states, actions):
    key1, key2 = jax.random.split(key)
    s_next, rew, terminals, _ = model(key1, states, actions)
    model_idxs = jax.random.choice(key2, model.apply_fn.elites, states.shape[:-1])
    s_next = jnp.take_along_axis(s_next, model_idxs[None, ..., None], axis=0)[0]
    rew = jnp.take_along_axis(rew, model_idxs[None, ...], axis=0)[0]
    terminals = jnp.take_along_axis(terminals, model_idxs[None, ...], axis=0)[0]
    return s_next, rew, terminals, None


# BC pretraining for policy
def update_actor_bc(actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            {"params": actor_params}, batch.observations, 0.0, training=True
        )
        a = get_deter(dist)

        actor_loss = jnp.mean((a - batch.actions) ** 2)
        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


# Maximize Q(s, a)
def onestep_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    model: Model,
    batch: Batch,
    discount: float,
    temperature: float,
    H: int,
    num_repeat: int,
) -> Tuple[Model, InfoDict]:
    N = batch.observations.shape[0]
    Robs = (
        batch.observations[:, None, :]
        .repeat(repeats=num_repeat, axis=1)
        .reshape(N * num_repeat, -1)
    )
    Ra = get_deter(actor(Robs))

    def calculate_gae_fwd(Robs, Ra, key0):
        states, actions, mask_weights, keys = [Robs], [Ra], [1.0], [key0]
        for i in range(H):
            rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4)
            keys.append(key0)
            s_next, rew, terminal, _ = run_model(rng1, model, states[i], actions[i])
            a_next = get_deter(actor(s_next))
            states.append(s_next)
            actions.append(a_next)
            mask_weights.append(mask_weights[i] * (1 - terminal))

        states = jnp.stack(states, axis=0)
        actions = jnp.stack(actions, axis=0)
        mask_weights = jnp.stack(mask_weights, axis=0)
        return states, actions, mask_weights

    keys = jax.random.split(key, N * num_repeat)
    vmap_fwd = lambda func: jax.vmap(func, in_axes=0, out_axes=1)
    states0, actions0, mask_weights0 = vmap_fwd(calculate_gae_fwd)(Robs, Ra, keys)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, states0)
        actions = get_deter(dist)
        actor_loss = -(mask_weights0 * critic(states0, actions)).mean()
        policy_std = dist.scale if hasattr(dist, "scale") else dist.distribution.scale

        return actor_loss, {
            "actor_loss": actor_loss,
            "policy_std": (policy_std * mask_weights0[:, :, None]).mean()
            / mask_weights0.mean(),
            "abs_actions": jnp.abs(actions0).mean(),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


# Maximize lambda-returns
def DPG_gae_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    model: Model,
    batch: Batch,
    discount: float,
    temperature: float,
    lamb: float,
    H: int,
    beta: float,
    num_repeat: int,
) -> Tuple[Model, InfoDict]:
    N = batch.observations.shape[0]
    Robs = (
        batch.observations[:, None, :]
        .repeat(repeats=num_repeat, axis=1)
        .reshape(N * num_repeat, -1)
    )
    Ra = get_deter(actor(Robs))

    def calculate_gae_fwd(Robs, Ra, key0):
        states, rewards, actions, mask_weights, keys = [Robs], [], [Ra], [1.0], [key0]
        q_rollout, q_values, ep_weights = [], [critic(Robs, Ra)], []
        for i in range(H):
            rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4)
            keys.append(key0)
            s_next, rew, terminal, _ = run_model(rng1, model, states[i], actions[i])
            a_next = get_deter(actor(s_next))
            penalty = get_penalty(rng3, critic, actor, model, states[i], actions[i])
            states.append(s_next)
            actions.append(a_next)
            rewards.append(rew - beta * penalty)
            mask_weights.append(mask_weights[i] * (1 - terminal))
            q_values.append(critic(s_next, a_next))

        q_rollout, lamb_weight = [q_values[-1]], 1.0
        for i in reversed(range(H)):
            q_next = (
                mask_weights[i] * rewards[i]
                + mask_weights[i + 1] * discount * q_rollout[-1]
            )
            next_value = (q_values[i] + lamb * lamb_weight * q_next) / (
                1 + lamb * lamb_weight
            )
            q_rollout.append(next_value)
            lamb_weight = 1.0 + lamb * lamb_weight
        q_rollout = list(reversed(q_rollout))

        states = jnp.stack(states, axis=0)
        actions = jnp.stack(actions, axis=0)
        mask_weights = jnp.stack(mask_weights, axis=0)
        q_rollout = jnp.stack(q_rollout, axis=0)
        return states, actions, mask_weights, q_rollout

    keys = jax.random.split(key, N * num_repeat)
    vmap_fwd = lambda func: jax.vmap(func, in_axes=0, out_axes=1)
    states0, actions0, mask_weights0, q_rollout = vmap_fwd(calculate_gae_fwd)(
        Robs, Ra, keys
    )

    def calculate_gae_bwd(delta_a, Robs, Ra, key0):
        states, rewards, actions, mask_weights, keys = (
            [Robs],
            [],
            [Ra + delta_a[0]],
            [1.0],
            [key0],
        )
        q_rollout, q_values = [], [critic(Robs, Ra + delta_a[0])]
        for i in range(H):
            rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4)
            keys.append(key0)
            s_next, rew, terminal, _ = run_model(rng1, model, states[i], actions[i])
            a_next = get_deter(actor(s_next)) + delta_a[i + 1]
            penalty = get_penalty(
                rng3, critic, actor, model, states[i], actions[i] + delta_a[i]
            )
            states.append(s_next)
            actions.append(a_next)
            rewards.append(rew - beta * penalty)
            mask_weights.append(mask_weights[i] * (1 - terminal))
            q_values.append(critic(s_next, a_next))

        q_rollout, lamb_weight = [q_values[-1]], 1.0
        for i in reversed(range(H)):
            q_next = (
                mask_weights[i] * rewards[i]
                + mask_weights[i + 1] * discount * q_rollout[-1]
            )
            next_value = (q_values[i] + lamb * lamb_weight * q_next) / (
                1 + lamb * lamb_weight
            )
            q_rollout.append(next_value)
            lamb_weight = 1.0 + lamb * lamb_weight
        q_rollout = list(reversed(q_rollout))
        return jnp.stack(q_rollout, axis=0)

    delta_a = jnp.zeros_like(actions0)
    vmap_bwd = lambda func: jax.vmap(func, in_axes=(1, 0, 0, 0), out_axes=1)
    grad_gae = vmap_bwd(jax.jacrev(calculate_gae_bwd))(delta_a, Robs, Ra, keys)
    grad_gae = jnp.stack([grad_gae[i, :, i] for i in range(H + 1)])

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, states0, 1.0, training=True)
        actions_grad = get_deter(dist)
        policy_std = dist.scale if hasattr(dist, "scale") else dist.distribution.scale
        actor_loss = -(grad_gae * actions_grad).mean(axis=1).sum()

        return actor_loss, {
            "actor_loss": actor_loss,
            "q_rollout": q_rollout.mean(),
            "policy_std": (policy_std * mask_weights0[:, :, None]).mean()
            / mask_weights0.mean(),
            "abs_actions": jnp.abs(actions0).mean(),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info
