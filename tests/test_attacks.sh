#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --job-name=TestAttacks
#SBATCH --output=slurm/TestAttacks.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4
#SBATCH --error=slurm/TestAttacks.out

echo TestAttacks

python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack pcfa --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack pcfa --attack_targeted True --attack_iterations 5
# python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack fab --attack_iterations 5
# python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack fab --attack_targeted True --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack common_corruptions
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack 3dcc
# python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack weather --attack_targeted True --attack_iterations 5



python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack bim --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack bim --attack_targeted True --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack fgsm --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack fgsm --attack_targeted True --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack apgd --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack apgd --attack_targeted True --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack cospgd --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack cospgd --attack_targeted True --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack pgd --attack_iterations 5
python -m flowbench.attacks --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack pgd --attack_targeted True --attack_iterations 5
