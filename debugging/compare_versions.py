import time
import torch

import subprocess
from loguru import logger
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

hostname = os.uname()[1].split('.')[0]

# not executable
# python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fab --attack_obj.attack_iterations 2
# python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fab --attack_obj.targeted True --attack_obj.attack_iterations 2

# checked and working
"""
"""

cmd = """
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pcfa --attack_obj.attack_iterations 2 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cc2d --attack_obj.cc_name gaussian_noise --attack_obj.cc_severity 1 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj bim --attack_obj.attack_iterations 2 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj bim --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj apgd --attack_obj.apgd_steps 5 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj apgd --attack_obj.targeted True --attack_obj.apgd_steps 5 --attack_obj.target zero --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cospgd --attack_obj.attack_iterations 2 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cospgd --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pgd --attack_obj.attack_iterations 2 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pgd --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fgsm --attack_obj.attack_iterations 2 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj fgsm --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj cc3d --attack_obj.tdcc_corruption far_focus --attack_obj.tdcc_intensity 1 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset sintel-final-trainval --attack_obj weather --attack_obj.targeted True --attack_obj.weather_steps 2 --attack_obj.weather_data /data2/katrin/datasets/sintel_weather_data/datasets/adv_weather_data/weather_rain_15 --max_samples 1
python -m flowbench.attacks_refactored --model raft --model.iters 2 --ckpt_path chairs --data.val_dataset kitti-2015-trainval --max_samples 1 --attack_obj none
"""
# check later
# 


commands = cmd.strip().split("\n")

for command in commands:

    args_new = command.split(" ") + ["--data.dataset_config_path", f"datasets_{hostname}.yaml"]

    def map_new_to_old(arg):
        if arg == "flowbench.attacks_refactored":
            arg = "flowbench.attacks"
        elif arg == "--attack_obj":
            arg = "--attack"
        elif arg == "--attack_obj.targeted":
            arg = "--attack_targeted"
        elif arg == "--attack_obj.target":
            arg = "--attack_target"
        elif arg == "cc2d":
            arg = "common_corruptions"
        elif arg == "cc3d":
            arg = "3dcc"
        else:
            arg = arg.replace("attack_obj.", "")
        return arg

    args_old = [
        map_new_to_old(arg)
        for arg in args_new
    ]
    print()
    print("Execute command $", " ".join(args_old))
    x = subprocess.run(
        [*args_old]
    )
    print("Execute command $", " ".join(args_new))
    y = subprocess.run(
        [*args_new]
    )

    print()
    print("Done with command $", command)
    print(x, y)
    print(x.returncode, y.returncode)
    
    if x.returncode != 0 or y.returncode != 0:
        logger.error("Have a look at this!")
        time.sleep(5)
