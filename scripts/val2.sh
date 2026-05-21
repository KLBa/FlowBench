#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:29:59
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --job-name=val2
#SBATCH --output=slurm/val2.out
#SBATCH --error=slurm/val2.out



cd ..

python -m flowbench.attacks --model ccmr --ckpt_path kitti --data.val_dataset kitti-2015
python -m flowbench.attacks --model craft --ckpt_path kitti --data.val_dataset kitti-2015
python -m flowbench.attacks --model csflow --ckpt_path kitti --data.val_dataset kitti-2015