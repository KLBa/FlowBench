#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --job-name=val4
#SBATCH --output=slurm/val4.out
#SBATCH --error=slurm/val4.out



cd ..

python -m flowbench.attacks --model maskflownet --ckpt_path kitti --data.val_dataset kitti-2015
python -m flowbench.attacks --model flow1d --ckpt_path kitti --data.val_dataset kitti-2015
python -m flowbench.attacks --model flowformer --ckpt_path kitti --data.val_dataset kitti-2015