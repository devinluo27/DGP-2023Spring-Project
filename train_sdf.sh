#!/bin/bash

source ~/.bashrc
source activate slotformer
alias python=python3
# export CPATH="$CPATH:$CONDA_PREFIX/include"

export config="runs/shapenet_chair138/srt/config.yaml"


export N_GPUS=1
export g0="0,1"
export g2="2"
export g3="2,3,6,7"
export port=$((RANDOM + 10000)) # $RANDOM(15bits) or shuf -i 2000-65000 -n 1

module load cudnn/8.6.0
module load cuda/11.7.1
module load gcc/10.2
module load gitlfs/2.7.1
# module load opengl/mesa-18.3.3
module load opengl/mesa-12.0.6 
module list

echo $CUDA_HOME

# CUDA_VISIBLE_DEVICES=$g0 python -m torch.distributed.launch \
CUDA_VISIBLE_DEVICES=$g0 \
torchrun --standalone --nnodes 1 --nproc_per_node $N_GPUS \
train.py $config

