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

python -c "from ptlflow.utils.dummy_datasets import write_kitti; write_kitti('./tmp/dummy')"
echo "kitti_2015: ./tmp/dummy/KITTI/2015" > ./datasets.yaml

python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pcfa --attack_obj.attack_iterations 5
# python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fab --attack_obj.attack_iterations 5
# python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fab --attack_obj.targeted True --attack_obj.attack_iterations 5
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cc2d --attack_obj.cc_name gaussian_noise --attack_obj.cc_severity 1
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cc3d --attack_obj.tdcc_corruption far_focus --attack_obj.tdcc_intensity 1
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj weather --attack_obj.targeted True --attack_obj.attack_iterations 5



python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj bim --attack_obj.attack_iterations 5
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj bim --attack_obj.targeted True --attack_obj.attack_iterations 5 --attack_obj.target zero
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fgsm --attack_obj.attack_iterations 5
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fgsm --attack_obj.targeted True --attack_obj.attack_iterations 5 --attack_obj.target zero
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj apgd --attack_obj.apgd_steps 5
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj apgd --attack_obj.targeted True --attack_obj.apgd_steps 5 --attack_obj.target zero
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cospgd --attack_obj.attack_iterations 5
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cospgd --attack_obj.targeted True --attack_obj.attack_iterations 5 --attack_obj.target zero
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pgd --attack_obj.attack_iterations 5
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pgd --attack_obj.targeted True --attack_obj.attack_iterations 5 --attack_obj.target zero
