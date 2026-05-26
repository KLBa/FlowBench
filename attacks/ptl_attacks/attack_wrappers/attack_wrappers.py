
import os
from typing import Any, Dict, Literal, Optional, Tuple, Union

from attacks.ptl_attacks.registry import register_attack
from ..base_attack.base_attack import BaseAttack

from ...fgsm import fgsm as fgsm_func
from ...apgd import apgd as apgd_func
from ...bim_pgd_cospgd import bim_pgd_cospgd as bim_pgd_cospgd_func
from ...fab import fab as fab_func
from ...common_corruptions import common_corrupt as common_corrupt_func
from ...weather import weather as weather_func


class FGSM(BaseAttack):

    def __init__(
        self, 
        attack_loss: Literal["epe"] = "epe",
        attack_norm: Literal["two", "inf"] = "inf",
        attack_epsilon: float = 8 / 255,
        attack_iterations: int = 3,
        targeted: bool = False,
        target: Optional[str] = None,

    ):
        super().__init__(targeted=targeted, target=target)
        self.attack_loss = attack_loss
        self.attack_norm = attack_norm
        self.attack_epsilon = attack_epsilon
        self.attack_iterations = attack_iterations

    def forward(self, inputs: Dict[str, Any]):
        preds, perturbed_inputs = fgsm_func(
            attack_args=self.__dict__,
            inputs=inputs['inputs'],
            model=inputs['model'],
            targeted_inputs=inputs['targeted_inputs'],
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs
        }

@register_attack
class fgsm(FGSM):
    pass


class BimPgdCospgd(BaseAttack):

    def __init__(
        self, 
        attack_loss: Literal["epe"] = "epe",
        targeted: bool = False,
        target: Optional[str] = None,
        attack_norm: Literal["two", "inf"] = "inf",
        attack_epsilon: float = 8 / 255,
        attack: Literal["bim", "pgd", "cospgd"] = "bim",
        attack_iterations: int = 3,
        attack_optim_target: Literal["ground_truth", "initial_flow"] = "ground_truth",
        attack_alpha: float = 0.01,
    ):
        super().__init__(targeted=targeted, target=target)
        self.attack_loss = attack_loss
        self.attack_targeted = targeted
        self.attack_norm = attack_norm
        self.attack_epsilon = attack_epsilon
        self.attack = attack
        self.attack_iterations = attack_iterations
        self.attack_optim_target = attack_optim_target
        self.attack_alpha = attack_alpha

    def forward(self, inputs: Dict[str, Any]):
        model = inputs['model']
        inputs_model = inputs['inputs']
        orig_preds = model(inputs_model)
        preds, perturbed_inputs, iteration_metrics = bim_pgd_cospgd_func(
            attack_args=self.__dict__,
            inputs=inputs_model,
            model=model,
            targeted_inputs=inputs['targeted_inputs'],
            orig_preds=orig_preds
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs,
            "iteration_metrics": iteration_metrics
        }


class BIM(BimPgdCospgd):
    def __init__(
        self, 
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            attack = "bim",
        )

@register_attack
class bim(BIM):
    pass


class CosPGD(BimPgdCospgd):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(
            **kwargs,
            attack = "cospgd",
        )

@register_attack
class cospgd(CosPGD):
    pass


class PGD(BimPgdCospgd):
    def __init__(
        self, 
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            attack = "pgd",
        )

@register_attack
class pgd(PGD):
    pass


class APGD(BaseAttack):
    
    def __init__(
        self, 
        attack_loss: Literal["epe"] = "epe",
        targeted: bool = False,
        target: Optional[str] = None,
        attack_norm: Literal["two", "inf"] = "inf",
        attack_epsilon: float = 8 / 255,
        apgd_steps: int = 10,
        apgd_n_restarts: int = 1,
        apgd_seed: int = 0,
        apgd_rho: float = 0.75,
        apgd_eot_iter: int = 1,
    ):
        super().__init__(targeted=targeted, target=target)
        self.attack_loss = attack_loss
        self.attack_targeted = targeted
        self.attack_norm = attack_norm
        self.attack_epsilon = attack_epsilon
        self.apgd_steps = apgd_steps
        self.apgd_n_restarts = apgd_n_restarts
        self.apgd_seed = apgd_seed
        self.apgd_rho = apgd_rho
        self.apgd_eot_iter = apgd_eot_iter

    def forward(self, inputs: Dict[str, Any]):
        model = inputs['model']
        inputs_model = inputs['inputs']
        preds, perturbed_inputs = apgd_func(
            attack_args=self.__dict__,
            inputs=inputs_model,
            model=model,
            targeted_inputs=inputs['targeted_inputs']
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs
        }

@register_attack
class apgd(APGD):
    pass


class FAB(BaseAttack):
    
    def __init__(
        self,
        attack_epsilon: float = 8 / 255,
        targeted: bool = False,
        attack_iterations: int = 3,
        target: Optional[str] = None,
    ):
        super().__init__(targeted=targeted, target=target)
        self.attack_targeted = targeted
        self.attack_epsilon = attack_epsilon
        self.attack_iterations = attack_iterations

    def forward(self, inputs: Dict[str, Any]):
        model = inputs['model']
        inputs_model = inputs['inputs']
        preds, perturbed_inputs = fab_func(
            attack_args=self.__dict__,
            inputs=inputs_model,
            model=model,
            targeted_inputs=inputs['targeted_inputs']
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs
        }

@register_attack
class fab(FAB):
    pass


class Weather(BaseAttack):
    bool_or_rgb = Union[bool, Tuple[int]]
    def __init__(
        self,
        attack_epsilon: float = 8 / 255,
        attack_loss: Literal["mse", "epe"] = "epe",
        weather_data: str = "./datasets/adv_weather_data",
        weather_scene_scale: float = 1.0,
        weather_recolor: bool_or_rgb = False,
        weather_learn_offset: bool_or_rgb = True,
        weather_learn_motionoffset: bool_or_rgb = True,
        weather_learn_transparency: bool_or_rgb = True,
        weather_learn_color: bool_or_rgb = True,
        weather_optimizer: Literal["Adam"] = "Adam",
        weather_steps: int = 750,
        weather_alph_motion: float = 1000.0,
        weather_alph_motionoffset: float = 1000.0,
        weather_do_motionblur: bool_or_rgb = True,
        weather_motionblur_samples: int = 10,
        weather_motionblur_scale: float = 0.025,
        weather_depth_check: bool_or_rgb = False,
        weather_depth_check_differentiable: bool_or_rgb = False,
        weather_transparency_scale: float = 1.0,
        weather_rendering_method: Literal["meshkin", "additive"] = "additive",
        weather_flakesize_max: int = 71,
        weather_constant_transparency: float = 0,
        
        # Arguments defined in attacks.py but unused by the attack
        # weather_depth_decay
        # weather_constant_transparency
        # weather_motion_y
        # weather_motion_random_scale
        # weather_motion_random_angle
        # weather_flake_r
        # weather_flake_g
        # weather_flake_b
        # weather_flake_random_h
        # weather_flake_random_l
        # weather_frame_per_scene
        # weather_no_flake_dat
        # weather_lr
        # weather_unregistered_artifacts
        # weather_num_flakes
        targeted: bool = True,
        target: Optional[str] = "zero"
    ):
        super().__init__(targeted=True, target=target)
        if not targeted:
            raise NotImplementedError("This weather attack implementation is a targeted attack.")

        if not os.path.exists(weather_data):
            raise ValueError(f"Folder {weather_data} does not exist. "
                             "Please set the path to the weather data with --attack_obj.weather_data /path/to/generated_weather_npz_data. "
                             "For more information, see <https://github.com/shashankskagnihotri/FlowBench#adversarial-weather>. ") 

        self.attack_epsilon = attack_epsilon
        self.attack_loss = attack_loss
        self.weather_data = weather_data
        self.weather_scene_scale = weather_scene_scale
        self.weather_recolor = weather_recolor
        self.weather_learn_offset = weather_learn_offset
        self.weather_learn_motionoffset = weather_learn_motionoffset
        self.weather_learn_transparency = weather_learn_transparency
        self.weather_learn_color = weather_learn_color
        self.weather_optimizer = weather_optimizer
        self.weather_steps = weather_steps
        self.weather_alph_motion = weather_alph_motion
        self.weather_alph_motionoffset = weather_alph_motionoffset
        self.weather_do_motionblur = weather_do_motionblur
        self.weather_motionblur_samples = weather_motionblur_samples
        self.weather_motionblur_scale = weather_motionblur_scale
        self.weather_depth_check = weather_depth_check
        self.weather_depth_check_differentiable = weather_depth_check_differentiable
        self.weather_transparency_scale = weather_transparency_scale
        self.weather_rendering_method = weather_rendering_method
        self.weather_flakesize_max = weather_flakesize_max
        self.weather_constant_transparency = weather_constant_transparency
        self.save_hyperparameters()

    def forward(self, inputs: Dict[str, Any]):
        model = inputs['model']
        inputs_model = inputs['inputs']
        targeted_inputs =  inputs['targeted_inputs']
        preds, perturbed_inputs = weather_func(
            attack_args=self.__dict__,
            model=model,
            targeted_inputs=targeted_inputs,
            batch_index=None,
            output_path=None,
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs
        }

@register_attack
class weather(Weather):
    pass


class CommonCorruptions(BaseAttack):
    def __init__(
        self,
        cc_name: str,
        cc_severity: int,
    ):
        super().__init__(targeted=False)
        self.cc_name = cc_name
        self.cc_severity = cc_severity

    def forward(self, inputs: Dict[str, Any]):
        model = inputs['model']
        inputs_model = inputs['inputs']
        preds, perturbed_inputs = common_corrupt_func(
            attack_args=self.__dict__,
            inputs=inputs_model,
            model=model,
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs
        }

@register_attack
class cc2d(CommonCorruptions):
    pass


class IdentityAttack(BaseAttack):
    def __init__(
        self,
    ):
        super().__init__(targeted=False)

    def forward(self, inputs: Dict[str, Any]):
        model = inputs['model']
        inputs_model = inputs['inputs']
        preds = model(inputs_model)
        perturbed_inputs = inputs_model.copy()
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs
        }

@register_attack
class none(IdentityAttack):
    pass
