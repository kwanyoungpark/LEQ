import os
from typing import Tuple

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"
os.environ["NUM_INTER_THREADS"] = "1"

import cv2
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import jax
import jax.numpy as jnp
import subprocess
import pickle as pkl
import wandb
import orbax.checkpoint

jax.config.update("jax_transfer_guard_device_to_host", "log")
jax.config.update("jax_transfer_guard_host_to_device", "log")
print("DISALLOW TRANSFERS")

from dynamics.termination_fns import get_termination_fn
import wrappers
from dataset_utils import (
    D4RLDataset,
    NeoRLDataset,
    split_into_trajectories,
    ReplayBuffer,
)
from evaluation import evaluate
from algos.mobileq.learner import Learner
import common
from common import log_info

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "antmaze-medium-play-v0", "Environment name.")
flags.DEFINE_string("load_dir", None, "Dynamics model load dir")
flags.DEFINE_string("save_dir", "./tmp/MOBILEQ/", "Tensorboard logging dir.")
flags.DEFINE_string("wandb_key", "", "Wandb key")
flags.DEFINE_string("dynamics", "torch", "Dynamics model")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("num_layers", 3, "number of hidden layers")
flags.DEFINE_integer("layer_size", 256, "layer size")
flags.DEFINE_integer("log_interval", 10000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("save_interval", 100000, "Save interval.")
flags.DEFINE_integer("video_interval", 50000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_float("discount", 0.997, "discount")
flags.DEFINE_float("lamb", 0.95, "lambda for GAE")
flags.DEFINE_float("beta", None, "Beta for LCB penalization")
flags.DEFINE_float("model_batch_ratio", 0.25, "Model-data weight ratio.")
flags.DEFINE_integer("rollout_batch_size", 50000, "Rollout batch size.")
flags.DEFINE_integer("rollout_freq", 1000, "Rollout batch size.")
flags.DEFINE_integer("rollout_length", 5, "Rollout length.")
flags.DEFINE_integer("num_repeat", 1, "Number of rollouts")
flags.DEFINE_integer("rollout_retain", 5, "Rollout retain")
flags.DEFINE_integer("horizon_length", 10, "Value estimation length.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("debug", False, "Debug (Not using wandb)")
flags.DEFINE_boolean("pretrain", False, "Use BC + FQE pretrain")
flags.DEFINE_boolean("no_rollout", False, "Do not use dataset expansion")
flags.DEFINE_boolean("maintain_model", False, "Maintain model while rollout")
flags.DEFINE_string("actor_update", "lambda-return", "Actor update method")
flags.DEFINE_string("critic_update", "lambda-return", "Critic update method")
config_flags.DEFINE_config_file(
    "config",
    "configs/config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def normalize(dataset):

    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    scale = 1000.0 / (compute_returns(trajs[-1]) - compute_returns(trajs[0]))
    dataset.rewards *= scale
    dataset.returns_to_go *= scale
    return scale, 0.0


def make_env_and_dataset(env_name, seed, discount, model=None):

    is_neorl = env_name.split("-")[1] == "v3"
    if is_neorl:
        import neorl
        import gym

        task, version, data_type = tuple(env_name.split("-"))
        env = neorl.make(task + "-" + version)
        dataset = NeoRLDataset(env, data_type, discount)
        raw_dataset = None
    else:
        import d4rl
        import d4rl_ext
        import gym

        env = gym.make(env_name)
        dataset = D4RLDataset(env, discount)
        raw_dataset = env.get_dataset()
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env_name = FLAGS.env_name.lower()
    if "antmaze" in env_name:
        # Following IQL, subtract 1.0 from rewards
        dataset.rewards -= 1.0
        reward_scale, reward_bias = 1.0, -1.0

    elif "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        if "random" in env_name:
            # For random datasets, use the original rewards
            # Because random trajectory is too bad, has low std.
            reward_scale, reward_bias = 1.0, 0.0
        else:
            # Otherwise, normalize the rewards same as IQL.
            reward_scale, reward_bias = normalize(dataset)
    else:
        reward_scale, reward_bias = 1.0, 0.0
    print("Reward scaler", reward_scale, reward_bias)
    return env, raw_dataset, dataset, (reward_scale, reward_bias)


def get_normalized_score_neorl(x, env_name):
    if env_name == "HalfCheetah":
        max_score = 12284
        min_score = -298
    if env_name == "Hopper":
        max_score = 3294
        min_score = 5
    if env_name == "Walker2d":
        max_score = 5143
        min_score = 1
    return (x - min_score) / (max_score - min_score)


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    kwargs = dict(FLAGS.config)

    print(FLAGS.flag_values_dict())
    if FLAGS.debug is False:
        wandb.login(key=FLAGS.wandb_key)
        run = wandb.init(
            # Set the project where this run will be logged
            project="IQL",
            name=f"MOBILEQ_{FLAGS.env_name}_{FLAGS.seed}",
            # Track hyperparameters and run metadata
            config={
                **FLAGS.flag_values_dict(),
                **kwargs,
            },
        )
        wandb.tensorboard.patch(save=False)
        wandb.define_metric("training/step")
        wandb.define_metric("training/*", step_metric="training/step")
        wandb.define_metric("evaluation/step")
        wandb.define_metric("evaluation/*", step_metric="evaluation/step")
    else:
        run = None

    if "dmc" in FLAGS.env_name:
        import gym

        _, task, diff = FLAGS.env_name.split("-")
        env = common.DMC(task, 2, (64, 64), -1)
        print(env.reset())
    else:
        if "v3" in FLAGS.env_name:
            import neorl
            import gym

            task, version, data_type = tuple(FLAGS.env_name.split("-"))
            env = neorl.make(task + "-" + version)
        else:
            import d4rl
            import d4rl_ext
            import gym

            env = gym.make(FLAGS.env_name)

    if FLAGS.dynamics == "torch":
        obs_dim, action_dim = (
            env.observation_space.shape[-1],
            env.action_space.shape[-1],
        )
        print(obs_dim, action_dim)
        termination_fn = get_termination_fn(task=FLAGS.env_name)
        if 1 <= FLAGS.seed and FLAGS.seed <= 5:
            print("TESTING SEEDS!")
            model_path = os.path.join(
                "../OfflineRL-Kit/models/dynamics-ensemble/",
                str(FLAGS.seed),
                FLAGS.env_name,
            )
        else:
            model_path = os.path.join(
                "../OfflineRL-Kit/models/dynamics-ensemble/", str(1), FLAGS.env_name
            )
        from dynamics.ensemble_model_learner import get_world_model

        env, raw_dataset, dataset, reward_scaler = make_env_and_dataset(
            FLAGS.env_name, FLAGS.seed, FLAGS.discount
        )
        with jax.transfer_guard("allow"):
            model, scaler = get_world_model(
                model_path, obs_dim, action_dim, reward_scaler, termination_fn, output_all=True
            )
        model_eval = None
    else:
        assert False, "Dynamics not given"

    if FLAGS.env_name.split("-")[1] == "v3":
        # NeoRL
        name, version, _ = FLAGS.env_name.split("-")
        env_name = name + "-" + version
        eval_envs = []
        for i in range(FLAGS.eval_episodes):
            env = gym.make(env_name, exclude_current_positions_from_observation=False)
            env.get_normalized_score = lambda x: get_normalized_score_neorl(x, name)
            env = wrappers.EpisodeMonitor(env)
            env = wrappers.SinglePrecision(env)
            seed = FLAGS.seed + i
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            eval_envs.append(env)
    else:
        # D4RL
        eval_envs = []
        for i in range(FLAGS.eval_episodes):
            env = gym.make(FLAGS.env_name)
            env = wrappers.EpisodeMonitor(env)
            env = wrappers.SinglePrecision(env)
            seed = FLAGS.seed + i
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            eval_envs.append(env)

    data_batch = dataset.sample(FLAGS.batch_size)
    print(data_batch.observations.shape)
    print(data_batch.actions.shape)
    print("Finished loading dataset")

    agent = Learner(
        FLAGS.seed,
        jax.device_put(data_batch.observations),
        jax.device_put(data_batch.actions),
        max_steps=FLAGS.max_steps,
        model=model,
        env_name=FLAGS.env_name,
        scaler=scaler,
        reward_scaler=reward_scaler,
        horizon_length=FLAGS.horizon_length,
        beta=FLAGS.beta,
        hidden_dims=tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)]),
        discount=FLAGS.discount,
        lamb=FLAGS.lamb,
        num_repeat=FLAGS.num_repeat,
        actor_update=FLAGS.actor_update,
        critic_update=FLAGS.critic_update,
        maintain_model=FLAGS.maintain_model,
        **kwargs,
    )

    rollout_dataset = ReplayBuffer(
        data_batch.observations.shape[1],
        data_batch.actions.shape[1],
        capacity=FLAGS.rollout_retain * FLAGS.rollout_length * FLAGS.rollout_batch_size,
    )

    key = common.PRNGKey(FLAGS.seed)
    if FLAGS.pretrain:
        if FLAGS.debug is False:
            wandb.define_metric("BC/step")
            wandb.define_metric("BC/*", step_metric="BC/step")
            wandb.define_metric("FQE/step")
            wandb.define_metric("FQE/*", step_metric="FQE/step")

        for i in tqdm.tqdm(range(200001)):
            data_batch = dataset.sample(4096)
            update_info = agent.update_bc(data_batch)

            if i % 50000 == 0:
                log_info(run, i, update_info, "BC")
        agent.actor = agent.actor.replace(params=agent.actor_pretrain.params)

        for i in tqdm.tqdm(range(200001)):
            data_batch = dataset.sample(FLAGS.batch_size)
            update_info = agent.update_fqe(data_batch)

            if i % 50000 == 0:
                log_info(run, i, update_info, "FQE")

        agent.critic = agent.critic.replace(params=agent.critic_pretrain.params)
        agent.target_critic = agent.target_critic.replace(
            params=agent.critic_pretrain.params
        )

    video_path = os.path.join(
        FLAGS.save_dir, "videos", FLAGS.env_name, str(FLAGS.seed), str(FLAGS.beta)
    )
    model_path = os.path.join(
        FLAGS.save_dir, "models", FLAGS.env_name, str(FLAGS.seed), str(FLAGS.beta)
    )
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    eval_stats = evaluate(FLAGS.seed, agent, eval_envs, "", 0, model_eval, debug=False)
    eval_stats = {f"average_{k}s": v for (k, v) in eval_stats.items()}
    log_info(run, 0, eval_stats, "evaluation")

    if FLAGS.debug:
        params = {"actor": agent.actor.params, "critic": agent.critic.params}
        with open(os.path.join(model_path, f"{FLAGS.seed}_0.pkl"), "wb") as F:
            pkl.dump(params, F)

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1):

        if FLAGS.no_rollout:
            data_batch = dataset.sample(FLAGS.batch_size)
            model_batch = dataset.sample(FLAGS.batch_size)
        else:
            if (i - 1) % FLAGS.rollout_freq == 0:
                key, rng = jax.random.split(key)
                data_batch = dataset.sample(FLAGS.rollout_batch_size)
                rollout = agent.rollout(
                    rng, data_batch.observations, FLAGS.rollout_length, 1.0
                )
                rollout_dataset.insert_batch(
                    rollout["obss"],
                    rollout["actions"],
                    rollout["rewards"],
                    rollout["masks"],
                    1 - rollout["masks"],
                    rollout["next_obss"],
                )

            data_batch = dataset.sample(FLAGS.batch_size)
            model_batch = rollout_dataset.sample(FLAGS.batch_size)

        update_info = agent.update(data_batch, model_batch, FLAGS.model_batch_ratio)

        if i % FLAGS.log_interval == 0:
            log_info(run, i, update_info, "training")

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(
                FLAGS.seed, agent, eval_envs, video_path, i, model_eval, debug=True
            )  # debug=FLAGS.debug)
            if raw_dataset is not None:
                obs = jax.device_put(raw_dataset["observations"][::10])
                act = jax.device_put(raw_dataset["actions"][::10])
                q_dataset = agent.critic(obs, act)
                q_dataset = jax.device_get(q_dataset)
                np.save(
                    os.path.join(video_path, f"q_values_dataset_{i}.npz"), q_dataset
                )

            print("Step", i, eval_stats["return"])
            eval_stats = {f"average_{k}s": v for (k, v) in eval_stats.items()}
            log_info(run, i, eval_stats, "evaluation")

        if i % FLAGS.save_interval == 0:
            params = {"actor": agent.actor.params, "critic": agent.critic.params}
            with open(os.path.join(model_path, f"{FLAGS.seed}_{i}.pkl"), "wb") as F:
                pkl.dump(params, F)

    score, length = [], []
    for i in range(10):
        eval_stats = evaluate(
            FLAGS.seed + i * FLAGS.eval_episodes,
            agent,
            eval_envs,
            video_path,
            1000000,
            model_eval,
        )
        score.append(eval_stats["return"])
        length.append(eval_stats["length"])
    run.log({f"evaluation/final_score": np.mean(score)}, step=1000000)
    run.log({f"evaluation/final_length": np.mean(length)}, step=1000000)


if __name__ == "__main__":
    app.run(main)
