#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --job-name=val6
#SBATCH --output=slurm/val6.out
#SBATCH --error=slurm/val6.out



cd ..

python -m flowbench.attacks --model irr --ckpt_path kitti --data.val_dataset kitti-2015
python -m flowbench.attacks --model llaflow --ckpt_path kitti --data.val_dataset kitti-2015
python -m flowbench.attacks --model rapidflow --ckpt_path kitti --data.val_dataset kitti-2015