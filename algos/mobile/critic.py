from typing import Tuple

import jax.numpy as jnp
import jax
import numpy as np

from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss

# MOBILE penalization
def get_penalty(
    key: PRNGKey,
    critic: Model,
    actor: Model,
    model: Model,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    num_repeat: jnp.ndarray,
):
    N = states.shape[0]
    key1, key2 = jax.random.split(key, 2)
    # Penalty calculation
    s_next, _, _, _ = model(key1, states, actions)  # [7, N, num_repeat]
    RNobs = s_next[:, None, :, :].repeat(repeats=num_repeat, axis=1)
    RNa = actor(RNobs).sample(seed=key2)
    q_next = critic(RNobs, RNa)
    penalty = q_next.mean(axis=1).std(axis=0)
    return penalty


# MOBILE q update
def update_q(
    key: PRNGKey,
    critic: Model,
    target_critic: Model,
    actor: Model,
    model: Model,
    data_batch: Batch,
    model_batch: Batch,
    discount: float,
    temperature: float,
    lamb: float,
    beta: float,
    num_repeat: int,
) -> Tuple[Model, Model, InfoDict]:

    key1, key2, key3, key4 = jax.random.split(key, 4)
    penalty = get_penalty(
        key2,
        critic,
        actor,
        model,
        model_batch.observations,
        model_batch.actions,
        num_repeat,
    )
    penalty = jnp.concatenate(
        [jnp.zeros((data_batch.observations.shape[0], *penalty.shape[1:])), penalty],
        axis=0,
    )

    obss = jnp.concatenate([data_batch.observations, model_batch.observations], axis=0)
    actions = jnp.concatenate([data_batch.actions, model_batch.actions], axis=0)
    rewards = jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0)
    masks = jnp.concatenate([data_batch.masks, model_batch.masks], axis=0)
    next_obss = jnp.concatenate(
        [data_batch.next_observations, model_batch.next_observations], axis=0
    )
    a_next = actor(next_obss).sample(seed=key1)

    next_value = critic(next_obss, a_next)
    target_q = (rewards - beta * penalty) + discount * masks * next_value
    # target_q = jnp.maximum(target_q, 0.)

    q_target = target_critic(obss, actions)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q = critic.apply({"params": critic_params}, obss, actions)
        critic_loss = loss(target_q, q, 0.5).mean()
        critic_reg_loss = ((q_target - q) ** 2).mean()

        critic_info = {
            "critic_loss": critic_loss,
            "critic_reg_loss": critic_reg_loss,
            "penalty": penalty.mean(),
            "discount": discount,
            "state_max": obss.max(),
            "action_max": actions.max(),
            "rewards": rewards.mean(),
            "rewards_min": rewards.min(),
            "rewards_max": rewards.max(),
            "masks": masks.mean(),
            "q": q.mean(),
            "q_min": q.min(),
            "q_max": q.max(),
        }

        return critic_loss + critic_reg_loss, critic_info

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, {**info}
