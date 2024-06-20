import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import gym
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import wandb

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'returns_to_go']
)

def PRNGKey(seed: int):
    return jax.device_put(np.array((seed, seed), dtype=np.uint32))

def expectile_loss(target, pred, expectile):
    weight = jnp.where(target > pred, expectile, (1 - expectile))
    diff = target - pred
    return weight * (diff ** 2)

def default_init(scale: Optional[float] = np.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def get_deter(dist):
    if hasattr(dist, 'scale'):
        return dist.mode()
    else:
        return jnp.tanh(dist.distribution.mode())

def get_stoch(dist, key, log_prob=False):
    if hasattr(dist, 'scale'):
        result = dist.sample(seed = key)
        if log_prob:
            log_probs = dist.log_prob(result)
    else:
        z = dist.distribution.sample(seed=key)
        result = jnp.tanh(z)
        if log_prob:
            log_probs = dist.distribution.log_prob(z) - (2. * (np.log(2.) - z - jax.nn.softplus(-2.*z)))
    if log_prob:
        return result, jnp.sum(log_probs, axis=-1)
    else:
        return result

Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    use_norm: Optional[bool] = False
    use_symlog: Optional[bool] = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.use_symlog: x = symlog(x)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_norm: x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

def log_info(run, step, update_info, prefix):
    if run is None:
        print("Step", step)
        for k, v in update_info.items():
            if v.ndim == 0:
                print(f'{prefix}/{k}:', jax.device_get(v))
            else:
                print("NOT SCALAR:", k)
    else:
        log_dict = {f'{prefix}/step': step}
        for k, v in update_info.items():
            if v.ndim == 0:
                log_dict[f'{prefix}/{k}'] = v
            else:
                log_dict[f'{prefix}/{k}'] = wandb.Histogram(v)
        run.log(log_dict)

@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def get_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        return grads, info

    def update_params(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grads, info = self.get_gradient(loss_fn)
        return self.update_params(grads), info

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
