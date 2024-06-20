from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    hidden_dims: Sequence[int]
    use_norm: bool = False
    use_symlog: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        observations = (observations - self.scaler[0]) / self.scaler[1]
        critic = MLP((*self.hidden_dims, 1), use_norm=self.use_norm, use_symlog=self.use_symlog)(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_norm: bool = False
    use_symlog: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        inputs = (inputs - self.scaler[0]) / self.scaler[1]
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations, use_norm=self.use_norm, use_symlog=self.use_symlog)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_norm: bool = False
    use_symlog: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.scaler, self.hidden_dims,
                         activations=self.activations,
                         use_norm=self.use_norm,
                         use_symlog=self.use_symlog)(observations, actions)
        critic2 = Critic(self.scaler, self.hidden_dims,
                         activations=self.activations,
                         use_norm=self.use_norm,
                         use_symlog=self.use_symlog)(observations, actions)
        return critic1, critic1
