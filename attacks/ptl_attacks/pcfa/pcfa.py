
from typing import Any, Dict, Literal
from ..base_attack.base_attack import BaseAttack

from ...pcfa import pcfa as pcfa_func
from ..registry import register_attack

class PCFA(BaseAttack):

    def __init__(
        self, 
        eps_box: float = 0.01, # also called attack_alpha
        attack_epsilon: float = 8 / 255,
        target: Literal["negative", "zero"] = "zero",
        pcfa_boxconstraint: Literal["clipping", "change_of_variables"] = "change_of_variables",
        attack_iterations: int = 3,
        attack_loss: str = "epe",
    ):
        super().__init__(targeted=True, target=target)
        self.eps_box = eps_box
        self.attack_epsilon = attack_epsilon
        self.pcfa_boxconstraint = pcfa_boxconstraint
        self.attack_iterations = attack_iterations
        self.attack_loss = attack_loss

    def attack(self, inputs: Dict[str, Any]):
        preds, perturbed_inputs, iteration_metrics = pcfa_func(
            attack_args=self.__dict__,
            inputs=inputs['inputs'],
            model=inputs['model'],
            targeted_inputs=inputs['targeted_inputs'],
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs,
            "iteration_metrics": iteration_metrics
        }


@register_attack
class pcfa(PCFA):
    pass
