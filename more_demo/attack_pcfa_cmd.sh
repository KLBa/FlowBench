python -m  flowbench.attacks \
        --model my_new_model.raft.my_very_own_raft \
        --model.iters 12 \
        --ckpt_path sintel \
        --attack pcfa \
        --attack_iterations 20 \
        --data.val_dataset sintel-final+kitti-2015 \
        --output_path outputs/attack
