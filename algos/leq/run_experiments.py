import vessl

start_command = 'mkdir ~/.mujoco &&' \
		'wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && ' \
		'tar -zxf mujoco.tar.gz -C "$HOME/.mujoco" && '\
		'rm mujoco.tar.gz && '\
		'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin && '\
		'pip install typing_extensions==4.9.0 && '\
		'pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl &&'\
		'pip install "Cython<3" && '\
		'pip install ml_collections && '\
		'pip install tensorboardX && '\
		'pip install tensorflow==2.13.0  && '\
		'pip install tensorflow_probability==0.21.0  && '\
		'pip install imageio && '\
		'pip install imageio-ffmpeg && '\
		'pip install wandb && '\
		'pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && '\
		'pip install typing_extensions==4.9.0 && '\
		'pip install numpy==1.23 && '\
		'pip install flax && '\
		'pip install optax==0.1.4 && '\
		'apt-get -qq update && '\
		'apt-get -qq install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf ffmpeg && '\
                'pip install opencv-python &&'\
                'git clone https://github.com/GGOSinon/OfflineRL-Kit.git && '\
                'git clone https://github.com/GGOSinon/implicit_q_learning.git && '\
		'pip install -e OfflineRL-Kit && '\
		'pip install typing_extensions==4.9.0 && '\
		'cd implicit_q_learning && '\
		'PYTHONPATH="." python train_myalgo_mopo.py --config $config --env_name $env_name --seed $seed --rollout_length $rollout_length --model_batch_ratio $model_batch_ratio --horizon_length 5 --expectile 0.1 --num_layers 3 --layer_size 256 --wandb_key $WANDB_KEY --video_interval 1000000'

rollout_lengths={'halfcheetah-random-v2': 5,
                 'hopper-random-v2': 5,
                 'walker2d-random-v2': 1,
                 'halfcheetah-medium-v2': 5,
                 'hopper-medium-v2': 5,
                 'walker2d-medium-v2': 1,
                 'halfcheetah-medium-replay-v2': 5,
                 'hopper-medium-replay-v2': 5,
                 'walker2d-medium-replay-v2': 1,
                 'halfcheetah-medium-expert-v2': 5,
                 'hopper-medium-expert-v2': 5,
                 'walker2d-medium-expert-v2': 1,
                }

model_batch_ratios={'halfcheetah-random-v2': 0.95,
                    'hopper-random-v2': 0.95,
                    'walker2d-random-v2': 0.95,
                    'halfcheetah-medium-v2': 0.95,
                    'hopper-medium-v2': 0.95,
                    'walker2d-medium-v2': 0.95,
                    'halfcheetah-medium-replay-v2': 0.95,
                    'hopper-medium-replay-v2': 0.95,
                    'walker2d-medium-replay-v2': 0.95,
                    'halfcheetah-medium-expert-v2': 0.5,
                    'hopper-medium-expert-v2': 0.5,
                    'walker2d-medium-expert-v2': 0.5,
                    }

             
env_names = ['halfcheetah-random-v2',
             'hopper-random-v2',
             'walker2d-random-v2',
             'halfcheetah-medium-v2',
             'hopper-medium-v2',
             'walker2d-medium-v2',
             'halfcheetah-medium-replay-v2',
             'hopper-medium-replay-v2',
             'walker2d-medium-replay-v2',
             'halfcheetah-medium-expert-v2',
             'hopper-medium-expert-v2',
             'walker2d-medium-expert-v2',
             ]

vessl.configure(
    organization_name='AI-RES-01',
    project_name='Offline-Model-based-RL',
)

for env_name in env_names:
    print(env_name)
    vessl.create_experiment(
        cluster_name="yonsei-ai-gpu",
        cluster_node_names=['ai-gpu-06', 'ai-gpu-08', 'ai-gpu-10', 'ai-gpu-12', 'ai-gpu-13', 'ai-gpu-14', 'ai-gpu-15'],
        kernel_resource_spec_name='gpu-1',
        kernel_image_url="quay.io/vessl-ai/ngc-tensorflow-kernel:22.12-tf1-py3-202301160809",
        start_command=start_command,
        hyperparameters=[f'env_name={env_name}',\
                         f'config=configs/myalgo/mujoco_config.py',\
                         f'seed=43',\
                         f'rollout_length={rollout_lengths[env_name]}',\
                         f'model_batch_ratio={model_batch_ratios[env_name]}',\
                        ],
        secrets=['WANDB_KEY=930383537a9a33cf8395f767553809f606c9bab0'],
    )

