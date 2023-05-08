#!/bin/bash

source ~/.bashrc
source activate slotformer
alias python=python3
# export CPATH="$CPATH:$CONDA_PREFIX/include"

# export config="runs/shapenet_chair138/srt/config_luotest.yaml"
# export config="runs/shapenet_chair138/srt_overfit_sin/config.yaml"
# export config="runs/shapenet_chair138/srt_overfit_sin_trunc_v2_run4_chair3/config.yaml"
export config="runs/shapenet_chair138/srt_overfit_sin_trunc_v2_run4_chair2/config.yaml"



export N_GPUS=1
export g0="0"
export g1="1"
export g2="2"
export g3="3"
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
# torchrun --standalone --nnodes 1 --nproc_per_node $N_GPUS \
CUDA_VISIBLE_DEVICES=$g1 \
python3 -m torch.distributed.launch \
--nproc_per_node=$N_GPUS --master_port=$port \
eval_sdf.py $config

