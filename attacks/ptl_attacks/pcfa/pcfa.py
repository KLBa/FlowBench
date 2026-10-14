
from typing import Any, Dict, Literal, Optional
from ..base_attack.base_attack import BaseAttack

from ...pcfa import pcfa as pcfa_func
from ..registry import register_attack

class PCFA(BaseAttack):

    def __init__(
        self, 
        attack_alpha: float = 1e-7,
        attack_epsilon: float = 0.005,
        target: Literal["negative", "zero"] = "zero",
        pcfa_boxconstraint: Literal["clipping", "change_of_variables"] = "change_of_variables",
        attack_iterations: int = 20,
        attack_loss: str = "epe",
        optim_mu: Optional[float] = None,
        optim_strong_wolfe: bool = False,
        best_iterate: bool = True,
    ):
        """
        Args:
            attack_alpha (float, optional): small numeric stabilization (`eps_box=1e-7` in the original PCFA code). Defaults to 1e-7.
            attack_epsilon (float, optional): bound on the average squared perturbation.
                The attack aims to find a perturbation delta with delta.pow(2).mean() < attack_epsilon.
                Defaults to 0.005 (as in the original PCFA).
            target (Literal[&quot;negative&quot;, &quot;zero&quot;], optional): Defaults to "zero".
            pcfa_boxconstraint (Literal[&quot;clipping&quot;, &quot;change_of_variables&quot;], optional): Defaults to "change_of_variables".
            attack_iterations (int, optional): Defaults to 20 (as in the original PCFA).
            attack_loss (str, optional): Defaults to "epe".
            optim_mu (Optional[float], optional): Factor for enforcing the delta bound. If None, mu is dynamically computed from epsilon. Defaults to None.
            optim_strong_wolfe (bool, optional): whether the LBFGS optimizer should use line_search_fn="strong_wolfe". Defaults to False.
            update_minima (bool, optional): If True, returns the perturbation with the smallest loss (original PCFA). If False, returns the last iteration (original FlowBench) 
        """

         # also called eps_box, should be set to 1e-6 but we preserve the FlowBench default for now.
        super().__init__(targeted=True, target=target)
        self.attack_alpha = attack_alpha
        self.attack_epsilon = attack_epsilon
        self.pcfa_boxconstraint = pcfa_boxconstraint
        self.attack_iterations = attack_iterations
        self.attack_loss = attack_loss
        self.optim_mu = optim_mu
        self.optim_strong_wolfe = optim_strong_wolfe
        self.best_iterate = best_iterate

        self.save_hyperparameters()

    def forward(self, inputs: Dict[str, Any]):
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
