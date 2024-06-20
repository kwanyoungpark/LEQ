# Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning

This repository contains the official implementation of [Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning](@TODO) by [Kwanyoung Park](https://ggosinon.github.io/) and [Youngwoon Lee](https://youngwoon.github.io/).

If you use this code for your research, please consider citing the paper:
```
@TODO
```

## How to run the code

### Install dependencies

```bash
conda create -n LEQ python=3.9

pip install -r requirements.txt

# Install jax
pip install jax[cuda]==0.4.8 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

# Install glew & others
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
export CPATH=$CONDA_PREFIX/include
pip install patchelf

# Recover versions
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip install numpy==1.23.0
pip install scipy==1.10.1
```
Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

Locomotion (D4RL)
```bash
PYTHONPATH='.' python train/train_LEQ.py --env_name=walker2d-medium-replay-v2 --expectile 0.5
```

Locomotion (D4RL)
```bash
PYTHONPATH='.' python train/train_LEQ.py --env_name=Hopper-v3-medium --expectile 0.1
```

AntMaze
```bash
PYTHONPATH='.' python train_offline.py --env_name=antmaze-large-play-v0 --expectile 0.3
```

## Misc
The implementation is based on [IQL](https://github.com/ikostrikov/implicit_q_learning/).
