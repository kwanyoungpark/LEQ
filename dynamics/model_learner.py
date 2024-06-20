"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from common import Batch, InfoDict, Model, PRNGKey, MLP, Params


@jax.jit
def _update_jit(
    rng: PRNGKey, model: Model, batch: Batch 
) -> Tuple[PRNGKey, Model, InfoDict]:

    s, a, r, sN, mask = batch.observations, batch.actions, batch.rewards[:, None], batch.next_observations, batch.masks[:, None]
    #jax.debug.print("r_hat:{r}, s_hat:{s}, 1-d:{x}", x=mask.mean(), r=r_hat.mean(), s=s_hat.mean())

    def loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        sN_hat, r_hat = model.apply({'params': params}, s, a)
        mu_s, logvar_s = sN_hat
        loss_rew = (((r_hat - r) ** 2)).mean()
        #loss_rew = jnp.exp(-logvar_r) * ((r - mu_r) ** 2) + logvar_r
        loss_dyn = jnp.exp(-logvar_s) * ((sN - mu_s) ** 2) + logvar_s
        loss_dyn = (mask * loss_dyn).mean()
        #loss_rep = (((s_hat - s) ** 2)).mean()
        loss = loss_dyn + loss_rew# + loss_rep
        return loss, {
            'lossR': loss_rew,
            'lossD': loss_dyn,
            'lossP': 0.,
        }

    new_model, info = model.apply_gradient(loss_fn)

    return rng, new_model, info
        
class _WorldModel(nn.Module):
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    dropout_rate: Optional[float] = None
    _min: Optional[float] = -10.
    _max: Optional[float] = 0.5

    def setup(self):
        self.encoder = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)
        self.decoder = MLP([self.hidden_dims[-1], self.obs_dim], activate_final=False, dropout_rate=self.dropout_rate)
        self.dynamics_head = MLP([self.hidden_dims[-1], self.hidden_dims[-1]], activate_final=False, dropout_rate=self.dropout_rate)
        self.reward_head = MLP([self.hidden_dims[-1], 1], activate_final=False, dropout_rate=self.dropout_rate)

    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        z = self.encoder(observations, training=training)
        zN_dist = self.dynamics_head(jnp.concatenate([z, actions], axis=1), training=training)
        mu, logvar = zN_dist[:, :self.hidden_dims[-1]], zN_dist[:, self.hidden_dims[-1]:]
        logvar = jnp.clip(logvar, self._min, self._max)

        s_hat = self.decoder(z, training=training)
        #sN_hat = self.decoder(zN, training=training)
        r_hat = self.reward_head(jnp.concatenate([z, actions], axis=1), training=training)

        return s_hat, (mu, logvar), r_hat

class WorldModel(nn.Module):
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    dropout_rate: Optional[float] = None
    _min: Optional[float] = -10.
    _max: Optional[float] = 0.5

    def setup(self):
        self.encoder = MLP(self.hidden_dims[:-1], activate_final=True, dropout_rate=self.dropout_rate)
        self.dynamics_head = MLP([self.hidden_dims[-1], self.obs_dim * 2], activate_final=False, dropout_rate=self.dropout_rate)
        self.reward_head = MLP([self.hidden_dims[-1], 1], activate_final=False, dropout_rate=self.dropout_rate)
        self.mask_head = MLP([self.hidden_dims[-1], 1], activate_final=False, dropout_rate=self.dropout_rate)

    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        z = self.encoder(jnp.concatenate([observations, actions], axis=1), training=training)

        zN_dist = self.dynamics_head(z, training=training)
        mu, logvar = zN_dist[:, :self.obs_dim], zN_dist[:, self.obs_dim:]
        logvar = jnp.clip(logvar, self._min, self._max)

        r_hat = self.reward_head(z, training=training).squeeze(1) 
        mask_hat = self.mask_head(z, training=training).squeeze(1)

        return (mu, logvar), r_hat, mask_hat

class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 lr: float = 3e-4,
                 model_hidden_dims: Sequence[int] = None,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        #print(dropout_rate, model_hidden_dims, max_steps)

        rng = jax.random.PRNGKey(seed)
        rng, model_key = jax.random.split(rng, 2)

        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        model_def = WorldModel(model_hidden_dims, obs_dim, action_dim, dropout_rate=dropout_rate)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=lr)

        model = Model.create(model_def,
                             inputs=[model_key, observations, actions],
                             tx=optimiser)

        self.model = model
        self.rng = rng

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_model, info = _update_jit(self.rng, self.model, batch)

        self.rng = new_rng
        self.model = new_model

        return info

