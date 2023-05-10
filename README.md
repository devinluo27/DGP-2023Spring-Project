# Brown ENG2501 2023 Srping Final Project

In this project, we aim to learn a neural network that takes images as input and outputs a corresponding mesh.

we use a backbone network based on SRT[1]. 

For more deatails, please refer to the [slides](https://docs.google.com/presentation/d/1ihUECglqh9L3kH-jz0aKBGTOO0TUBvqEoZ90e_3d2s0/).


## Running Experiments
Each run's config, checkpoints, and visualization are stored in a dedicated directory. Recommended configs can be found under `runs/[dataset]/[model]`.

### Training
To train a model on a single GPU, simply run e.g.:
```
python3 -m torch.distributed.launch \
--nproc_per_node=$N_GPUS --master_port=$port train_sdf.py $config
```

### evaluation
To evaluate a model:
```
python3 -m torch.distributed.launch \
--nproc_per_node=$N_GPUS --master_port=$port eval_sdf.py $config
```

## Reference
[1]Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations

