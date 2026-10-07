import subprocess
import pytest

from ptlflow.utils.dummy_datasets import write_kitti

write_kitti('./tmp/dummy', img_size=(97, 64))

shared_args = "--model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --data.kitti_2015_root_dir ./tmp/dummy/KITTI/2015".split(" ")
args_list = [
    "--attack_obj pcfa --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj fab --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj fab --attack_obj.targeted True --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj cc2d --attack_obj.cc_name gaussian_noise --attack_obj.cc_severity 1".split(" "),
    "--attack_obj cc3d --attack_obj.tdcc_corruption far_focus --attack_obj.tdcc_intensity 1".split(" "),
    "--attack_obj weather --attack_obj.targeted True --attack_obj.attack_iterations 2".split(" "),
    "".split(" "),
    "--attack_obj bim --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj bim --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero".split(" "),
    "--attack_obj fgsm --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj fgsm --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero".split(" "),
    "--attack_obj apgd --attack_obj.apgd_steps 5".split(" "),
    "--attack_obj apgd --attack_obj.targeted True --attack_obj.apgd_steps 5 --attack_obj.target zero".split(" "),
    "--attack_obj cospgd --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj cospgd --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero".split(" "),
    "--attack_obj pgd --attack_obj.attack_iterations 2".split(" "),
    "--attack_obj pgd --attack_obj.targeted True --attack_obj.attack_iterations 2 --attack_obj.target zero".split(" "),
]

@pytest.mark.parametrize("args", args_list)
def test_attack(args):
    result = subprocess.run(
        ["python", "-m", "flowbench.attacks_refactored", *shared_args, *args],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "config saved to" in result.stdout.lower()
