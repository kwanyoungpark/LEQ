## This code is the direct reimplementation of the world model in OfflineRLKit
## (https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/dynamics/ensemble_dynamics.py)

from typing import Optional, Sequence, Tuple, Callable

from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from dynamics.model_learner import WorldModel
from common import Batch, InfoDict, Model, PRNGKey, MLP, Params

def softplus(x):
    return jnp.logaddexp(x, 0)


def soft_clamp(x, _min, _max):
    x = _max - softplus(_max - x)
    x = _min + softplus(x - _min)
    return x


class EnsembleLinear(nn.Module):
    input_dim: int
    output_dim: int
    num_ensemble: int
    use_norm: bool = False

    def setup(self):
        self.weight = self.param(
            "kernel",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, self.input_dim, self.output_dim),
        )
        self.bias = self.param(
            "bias",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, 1, self.output_dim),
        )
        if self.use_norm:
            self.norm = nn.LayerNorm(epsilon=1e-05)

    def __call__(self, x: jnp.ndarray):
        x = jnp.einsum("nbi,nij->nbj", x, self.weight)
        if self.use_norm:
            x = self.norm(x)
        else:
            x = x + self.bias
        return x

class EnsembleDynamicModel(nn.Module):
    model: nn.Module
    elites: Tuple[int]
    terminal_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    output_all: bool = False
    clip_extremes: bool = False
    reward_mean: bool = False

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.ones,
            (2, self.model.obs_dim + self.model.action_dim),
        )
        self.reward_scaler = self.param("reward_scaler", nn.initializers.zeros, (2,))

    def __call__(
        self,
        key: PRNGKey,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        training: bool = False,
        model_idxs: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        key1, key2 = jax.random.split(key)
        shapes = observations.shape[:-1]
        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])

        z = jnp.concatenate([observations, actions], axis=1)
        z = (z - self.scaler[0]) / self.scaler[1]
        mean, logvar = self.model(z)
        next_obs = mean[:, :, :-1] + observations[None, :, :]
        mean = jnp.concatenate([next_obs, mean[:, :, -1:]], axis=2)
        std = jnp.sqrt(jnp.exp(logvar))

        ensemble_samples = mean + jax.random.normal(key1, mean.shape) * std

        num_models, batch_size, _ = ensemble_samples.shape

        if self.output_all:
            samples = ensemble_samples
            next_obs = samples[..., :-1]
            reward = samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = jax.vmap(self.terminal_fn, in_axes=(None, None, 0), out_axes=0)(
                observations, actions, next_obs
            ).squeeze(-1)
            print(terminal.shape)
            shapes = (samples.shape[0], *shapes)
            next_obs = next_obs.reshape((*shapes, -1))
            reward = reward.reshape(shapes)
            terminal = terminal.reshape(shapes)
        else:
            if model_idxs is None:
                model_idxs = jax.random.choice(key2, self.elites, (1, batch_size, 1))
            if self.reward_mean:
                next_obs = jnp.take_along_axis(
                    ensemble_samples[..., :-1], model_idxs, axis=0
                )[0]
                reward = (
                    ensemble_samples[..., -1].mean(axis=0) * self.reward_scaler[0]
                    + self.reward_scaler[1]
                )
            else:
                samples = jnp.take_along_axis(ensemble_samples, model_idxs, axis=0)[0]
                next_obs = samples[..., :-1]
                reward = (
                    samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
                )
            terminal = self.terminal_fn(observations, actions, next_obs).squeeze(1)
            next_obs = next_obs.reshape((*shapes, -1))
            reward = reward.reshape(shapes)
            terminal = terminal.reshape(shapes)

        if self.clip_extremes:
            obs_scaler = self.scaler[:, : self.model.obs_dim]
            next_obs = jnp.clip(
                next_obs,
                obs_scaler[0] - 15.0 * obs_scaler[1],
                obs_scaler[0] + 15.0 * obs_scaler[1],
            )
            z = (next_obs - obs_scaler[0]) / obs_scaler[1]
        else:
            pass
        info = {}
        info["raw_reward"] = reward

        return next_obs, reward, terminal, info


class EnsembleWorldModel(nn.Module):
    num_models: int
    num_elites: int
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    dropout_rate: Optional[float] = None
    use_norm: bool = False

    def setup(self):
        hidden_dims = (self.obs_dim + self.action_dim,) + self.hidden_dims
        self.layers = [
            EnsembleLinear(
                hidden_dims[i - 1], hidden_dims[i], self.num_models, self.use_norm
            )
            for i in range(1, len(hidden_dims))
        ]
        self.last_layer = EnsembleLinear(
            self.hidden_dims[-1], 2 * (self.obs_dim + 1), self.num_models
        )
        self.min_logvar = self.param(
            "min_logvar", nn.initializers.zeros, (self.obs_dim + 1,)
        )
        self.max_logvar = self.param(
            "max_logvar", nn.initializers.zeros, (self.obs_dim + 1,)
        )

    def __call__(
        self,
        z: jnp.ndarray,
        # actions: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # z = jnp.concatenate([observations, actions], axis=1)
        if len(z.shape) == 2:
            z = z[None, :, :].repeat(self.num_models, axis=0)
        for layer in self.layers:
            z = layer(z)
            z = nn.swish(z)
        z = self.last_layer(z)
        # mean, logvar, reward = jnp.split(z, [self.obs_dim, 2*self.obs_dim], axis=1)
        mean, logvar = z[:, :, : self.obs_dim + 1], z[:, :, self.obs_dim + 1 :]
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

        return mean, logvar

    def set_elites(self, metrics):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        idxs = np.sort(valid_losses)
        self.model.elites = idxs


import torch
import os


def get_world_model(
    model_path,
    obs_dim,
    action_dim,
    reward_scaler,
    termination_fn,
    deterministic=False,
    output_all=False,
    clip_extremes=False,
):
    observations = jax.device_put(np.zeros((1, obs_dim)))
    actions = jax.device_put(np.zeros((1, action_dim)))

    ### Define world model ###
    mu = np.load(os.path.join(model_path, "mu.npy"))
    std = np.load(os.path.join(model_path, "std.npy"))
    ckpt = torch.load(
        os.path.join(model_path, "dynamics.pth"), map_location=torch.device("cpu")
    )
    ckpt = {k: v.cpu().numpy() for (k, v) in ckpt.items()}
    elites = ckpt["elites"]
    scaler = (np.array(mu), np.array(std))
    scaler = jax.device_put(scaler)
    reward_scaler = np.stack(reward_scaler, axis=0)
    reward_scaler = jax.device_put(reward_scaler)
    model_def = EnsembleWorldModel(
        7, 5, (200, 200, 200, 200), obs_dim, action_dim, dropout_rate=None
    )
    model_def = EnsembleDynamicModel(
        model_def,
        elites,
        termination_fn,
        output_all=output_all,
        clip_extremes=clip_extremes,
    )

    model_key = PRNGKey(42)  
    model = Model.create(
        model_def, inputs=[model_key, model_key, observations, actions], tx=None
    )

    ### Load model parameters trained from OfflineRL-Kit ###
    ckpt_jax = {}
    for i in range(4):
        ckpt_jax[f"layers_{i}"] = {}
        ckpt_jax[f"layers_{i}"]["kernel"] = ckpt[f"backbones.{i}.weight"]
        ckpt_jax[f"layers_{i}"]["bias"] = ckpt[f"backbones.{i}.bias"]
    ckpt_jax[f"last_layer"] = {}
    ckpt_jax[f"last_layer"]["kernel"] = ckpt[f"output_layer.weight"]
    ckpt_jax[f"last_layer"]["bias"] = ckpt[f"output_layer.bias"]
    ckpt_jax["min_logvar"] = ckpt["min_logvar"]
    ckpt_jax["max_logvar"] = ckpt["max_logvar"]
    ckpt_jaxs = {"model": ckpt_jax}
    ckpt_jaxs["scaler"] = jnp.concatenate(scaler, axis=0)
    ckpt_jaxs["reward_scaler"] = jnp.stack(reward_scaler, axis=0)
    ckpt_jaxs["elites"] = elites
    ckpt_jaxs = jax.tree_util.tree_map(lambda x: jax.device_put(x), ckpt_jaxs)
    model = model.replace(params=ckpt_jaxs)

    return model, scaler
