import collections
from typing import Optional, Tuple
import glob, os

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from common import Batch, Model, PRNGKey


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        returns_to_go: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.returns_to_go = returns_to_go
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            returns_to_go=self.returns_to_go[indx],
        )


class NeoRLDataset(Dataset):
    def __init__(
        self,
        env,
        data_type: str,
        discount: float = 1.0,
        traj_num: int = 1000,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        train_data, _ = env.get_dataset(
            data_type=data_type, train_num=traj_num, need_val=False
        )
        dataset = {}
        dataset["observations"] = train_data["obs"]
        dataset["actions"] = train_data["action"]
        dataset["next_observations"] = train_data["next_obs"]
        dataset["rewards"] = train_data["reward"][:, 0]
        dataset["terminals"] = train_data["done"][:, 0]

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        returns_to_go = np.zeros_like(dataset["rewards"])
        # returns_to_go[-1] = dataset['rewards'][-1]
        # for i in reversed(range(len(dones_float) - 1)):
        #    returns_to_go[i] = dataset['rewards'][i] + returns_to_go[i+1] * discount * (1.0 - dones_float[i])

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            returns_to_go=returns_to_go.astype(np.float32),
            size=len(dataset["observations"]),
        )


class D4RLDataset(Dataset):
    def __init__(
        self, env, discount: float = 1.0, clip_to_eps: bool = True, eps: float = 1e-5
    ):
        import d4rl
        import gym

        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        returns_to_go = np.zeros_like(dataset["rewards"])
        # returns_to_go[-1] = dataset['rewards'][-1]
        # for i in reversed(range(len(dones_float) - 1)):
        #    returns_to_go[i] = dataset['rewards'][i] + returns_to_go[i+1] * discount * (1.0 - dones_float[i])

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            returns_to_go=returns_to_go.astype(np.float32),
            size=len(dataset["observations"]),
        )


class ReplayBuffer(Dataset):
    def __init__(self, observation_dim: int, action_dim: int, capacity: int):

        observations = np.empty((capacity, observation_dim), dtype=np.float32)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        returns_to_go = np.empty((capacity,), dtype=np.float32)
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            returns_to_go=returns_to_go,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]
        self.returns_to_go[:num_samples] = dataset.returns_to_go[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        returns_to_go: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation
        self.returns_to_go[self.insert_index] = returns_to_go

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def insert_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
    ):
        batch_size = observations.shape[0]
        if self.insert_index + batch_size > self.capacity:
            p = self.capacity - self.insert_index
            self.insert_batch(
                observations[:p],
                actions[:p],
                rewards[:p],
                masks[:p],
                dones_float[:p],
                next_observations[:p],
            )
            self.insert_batch(
                observations[p:],
                actions[p:],
                rewards[p:],
                masks[p:],
                dones_float[p:],
                next_observations[p:],
            )
            return
        self.observations[self.insert_index : self.insert_index + batch_size] = (
            observations
        )
        self.actions[self.insert_index : self.insert_index + batch_size] = actions
        self.rewards[self.insert_index : self.insert_index + batch_size] = rewards
        self.masks[self.insert_index : self.insert_index + batch_size] = masks
        self.dones_float[self.insert_index : self.insert_index + batch_size] = (
            dones_float
        )
        self.next_observations[self.insert_index : self.insert_index + batch_size] = (
            next_observations
        )
        self.returns_to_go[self.insert_index : self.insert_index + batch_size] = None

        self.insert_index = (self.insert_index + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
