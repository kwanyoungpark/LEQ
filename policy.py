import functools
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from common import MLP, Params, PRNGKey, default_init, Model

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class SACalpha(nn.Module):
    init_value: int = 0.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param(
            "log_alpha", nn.initializers.constant(self.init_value), ()
        )
        return log_alpha


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    observation_dim: int
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    use_norm: bool = True
    use_symlog: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        scaler = self.param("scaler", nn.initializers.zeros, (2, self.observation_dim))
        batch_shape = observations.shape[:-1]
        scaler = scaler.reshape((2,) + (1,) * len(batch_shape) + (-1,))
        mu, std = jnp.split(scaler, 2, axis=0)
        mu, std = mu.squeeze(0), std.squeeze(0)

        observations = (observations - mu) / std
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            use_norm=self.use_norm,
            use_symlog=self.use_symlog,
        )(observations, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.log_std_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
            log_stds = log_stds.reshape((1,) * len(batch_shape) + (self.action_dim,))

        # jax.debug.print("{x} {y}", x=means[0], y=log_stds[0])
        # print("LOG_STD_SHAPE", batch_shape, log_stds.shape, means.shape)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)
            log_stds = jnp.log(0.9 * jax.nn.sigmoid(log_stds + 2.0) + 0.1)
        else:
            log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        # base_dist = tfd.MultivariateNormalDiag(loc=means,
        #                                       scale_diag=jnp.exp(log_stds) *
        #                                       temperature)

        base_dist = tfd.Normal(loc=means, scale=jnp.exp(log_stds) * temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(
                distribution=base_dist, bijector=tfb.Tanh()
            )
        else:
            return base_dist


@functools.partial(jax.jit, static_argnames=("actor_def"))
def _sample_actions(
    rng: PRNGKey,
    actor_def: Model,
    actor_params: Params,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({"params": actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(
    rng: PRNGKey, actor: Model, observations: np.ndarray, temperature: float = 1.0
) -> Tuple[PRNGKey, jnp.ndarray]:
    observations = jax.device_put(observations)
    return _sample_actions(rng, actor.apply_fn, actor.params, observations, temperature)
