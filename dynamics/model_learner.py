## This code is the direct reimplementation of the world model in OfflineRLKit
## (https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/dynamics/ensemble_dynamics.py)

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from common import Batch, InfoDict, Model, PRNGKey, MLP, Params

class WorldModel(nn.Module):
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    dropout_rate: Optional[float] = None
    _min: Optional[float] = -10.0
    _max: Optional[float] = 0.5

    def setup(self):
        self.encoder = MLP(
            self.hidden_dims[:-1], activate_final=True, dropout_rate=self.dropout_rate
        )
        self.dynamics_head = MLP(
            [self.hidden_dims[-1], self.obs_dim * 2],
            activate_final=False,
            dropout_rate=self.dropout_rate,
        )
        self.reward_head = MLP(
            [self.hidden_dims[-1], 1],
            activate_final=False,
            dropout_rate=self.dropout_rate,
        )
        self.mask_head = MLP(
            [self.hidden_dims[-1], 1],
            activate_final=False,
            dropout_rate=self.dropout_rate,
        )

    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        z = self.encoder(
            jnp.concatenate([observations, actions], axis=1), training=training
        )

        zN_dist = self.dynamics_head(z, training=training)
        mu, logvar = zN_dist[:, : self.obs_dim], zN_dist[:, self.obs_dim :]
        logvar = jnp.clip(logvar, self._min, self._max)

        r_hat = self.reward_head(z, training=training).squeeze(1)
        mask_hat = self.mask_head(z, training=training).squeeze(1)

        return (mu, logvar), r_hat, mask_hat



