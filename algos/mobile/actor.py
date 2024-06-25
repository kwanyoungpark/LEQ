from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss

def update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
    discount: float,
    temperature: float,
    sac_alpha: float,
) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": key},
        )
        a = dist.sample(seed=key)
        log_probs = dist.log_prob(a)

        q = critic(batch.observations, a)

        actor_loss = -q.mean() + sac_alpha * log_probs.mean()

        policy_std = dist.scale if hasattr(dist, "scale") else dist.distribution.scale
        return actor_loss, {
            "actor_loss": actor_loss,
            "policy_std": policy_std.mean(),
            "log_probs": log_probs.mean(),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def update_alpha(
    key: PRNGKey, log_probs: jnp.ndarray, sac_alpha: Model, target_entropy: float
) -> Tuple[Model, InfoDict]:
    log_probs = log_probs + target_entropy

    def alpha_loss_fn(alpha_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_alpha = sac_alpha.apply({"params": alpha_params})
        alpha_loss = -(log_alpha * log_probs).mean()
        return alpha_loss, {"alpha_loss": alpha_loss, "alpha": jnp.exp(log_alpha)}

    new_alpha, info = sac_alpha.apply_gradient(alpha_loss_fn)

    return new_alpha, info
