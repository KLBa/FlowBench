from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
import torch
from torch import optim

from ptlflow.models.base_model.base_model import BaseModel

from ..base_attack.base_attack import BaseAttack

from ...attack_utils.utils import (
    get_image_tensors,
    replace_images_dic,
    get_flow_tensors,
)
from ...attack_utils  import loss_criterion as losses
from ...attack_utils.loss_criterion import get_loss

from ...pcfa import InputModel
from ..registry import register_attack

class PCFA2(BaseAttack):
    """Main difference to PCFA is that we optimize the perturbation `delta` instead of the perturbed image `nw_input`.
    This allows us to paramterize the delta differently, e.g., enforcing the delta bound by parametrization.
    """

    def __init__(
        self,
        attack_alpha: float = 1e-7,
        attack_epsilon: float = 8 / 255,
        target: Literal["negative", "zero"] = "zero",
        pcfa_boxconstraint: Literal["clipping", "change_of_variables"] = "clipping",
        pcfa_deltaconstraint: Literal["project", "homogeneous_scale"] = "project",
        attack_iterations: int = 3,
        attack_loss: str = "epe",
        optim_mu: Optional[float] = None,
        optim_strong_wolfe: bool = False,
    ):
        # also called eps_box, should be set to 1e-6 but we preserve the FlowBench default for now.
        super().__init__(targeted=True, target=target)
        self.attack_alpha = attack_alpha
        self.attack_epsilon = attack_epsilon
        self.pcfa_boxconstraint = pcfa_boxconstraint
        self.pcfa_deltaconstraint = pcfa_deltaconstraint
        self.attack_iterations = attack_iterations
        self.attack_loss = attack_loss
        self.optim_mu = optim_mu
        self.optim_strong_wolfe = optim_strong_wolfe

        if optim_mu:
            raise ValueError("optim_mu is deprecated")
        if pcfa_boxconstraint == "change_of_variables":
            raise NotImplementedError()

        self.save_hyperparameters()


    def forward(self, inputs: Dict[str, Any]):
        preds, perturbed_inputs, iteration_metrics = _pcfa2(
            attack_args=self.__dict__,
            inputs=inputs["inputs"],
            model=inputs["model"],
            targeted_inputs=inputs["targeted_inputs"],
        )
        return {
            "preds": preds,
            "perturbed_inputs": perturbed_inputs,
            "iteration_metrics": iteration_metrics,
        }


@register_attack
class pcfa2(PCFA2):
    pass


def _pcfa2(
    attack_args: Dict[str, Any],
    inputs: Dict[str, torch.Tensor],
    model: BaseModel,
    targeted_inputs: Dict[str, torch.Tensor],
):
    """
    Performs an PCFA attack on a given model and for all images of a specified dataset.
    """

    if attack_args.get("optim_mu", None) is not None:
        optim_mu = attack_args["optim_mu"]
    else:
        optim_mu = 2500.0 / attack_args["attack_epsilon"]

        if attack_args["attack_epsilon"] == 0.01:
            optim_mu = 100000
        elif attack_args["attack_epsilon"] == 0.001:
            optim_mu = 1000000

        if attack_args["attack_target"] not in ["zero"]:
            optim_mu = 1.5 * optim_mu

    eps_box = attack_args["attack_alpha"]

    # Define what device we are using
    device = model.device
    # if not torch.cuda.is_available():
    #   device = torch.device("cpu")
    # else:
    #   device = torch.device("cuda")

    # Make sure the model is not trained:
    for param in model.parameters():
        param.requires_grad = False

    preds, delta1, delta2, iteration_metrics = _pcfa_attack2(
        model, targeted_inputs, inputs, eps_box, device, optim_mu, attack_args
    )
    image1, image2 = get_image_tensors(targeted_inputs, clone=True)
    perturbed_image1 = image1 + delta1
    perturbed_image2 = image2 + delta2
    perturbed_inputs = replace_images_dic(
        targeted_inputs, perturbed_image1, perturbed_image2, clone=True
    )
    
    delta_norm_squared = losses.two_norm_avg_delta_squared(delta1, delta2)
    if delta_norm_squared > attack_args["attack_epsilon"] ** 2:
        logger.warning(f"Perturbation exceeds bound {delta_norm_squared.sqrt()} > {attack_args['attack_epsilon']}")

    return preds, perturbed_inputs, iteration_metrics


def _pcfa_attack2(
    model, targeted_inputs, inputs, eps_box, device, optim_mu, attack_args
):
    """Subroutine to optimize a PCFA perturbation on a given image pair. For a specified number of steps."""
    # For logging of different norms for the deltas and EPE for each iteration
    iteration_metrics = {}

    model = InputModel(
        model,
        eps_box,
        variable_change=attack_args["pcfa_boxconstraint"] in ["change_of_variables"],
    )

    image1, image2 = get_image_tensors(targeted_inputs)
    image1, image2 = image1.to(device), image2.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False

    # initialize perturbation and auxiliary variables:
    nw_delta1 = torch.zeros_like(image1)
    nw_delta2 = torch.zeros_like(image2)
    nw_delta1 = nw_delta1.to(device)
    nw_delta2 = nw_delta2.to(device)
    
    homogeneous = torch.tensor(1., device=device, requires_grad=True)

    nw_input1 = None
    nw_input2 = None

    # Set up the optimizer and variables if individual perturbations delta1 and delta2 for images 1 and 2 should be trained
    nw_delta1.requires_grad = True
    nw_delta2.requires_grad = True

    delta1, delta2 = transform_delta(
        nw_delta1, nw_delta2, attack_args["attack_epsilon"], homogeneous=homogeneous, pcfa_deltaconstraint=attack_args["pcfa_deltaconstraint"]
    )
    if attack_args["pcfa_boxconstraint"] in ["change_of_variables"]:
        nw_input1 = torch.atanh(
            2.0 * (1.0 - eps_box) * (image1 + delta1) - (1 - eps_box)
        )
        nw_input2 = torch.atanh(
            2.0 * (1.0 - eps_box) * (image2 + delta2) - (1 - eps_box)
        )
    else:
        nw_input1 = image1 + delta1
        nw_input2 = image2 + delta2

    # nw_input1.requires_grad = True
    # nw_input2.requires_grad = True

    params = [nw_delta1, nw_delta2]
    if attack_args["pcfa_deltaconstraint"] in ["homogeneous_scale"]:
        params.append(homogeneous)
    if attack_args.get("optim_strong_wolfe", False) is True:
        optimizer = optim.LBFGS(
            params, max_iter=10, line_search_fn="strong_wolfe"
        )
    else:
        # This is the original version but it may massively overshoot delta bound!
        optimizer = optim.LBFGS(params, max_iter=10)

    # define target (potentially based on first flow prediction)
    target = get_flow_tensors(targeted_inputs)
    target = target.to(device)
    target.requires_grad = False

    # save nw_inputs, images, flow_pred, target
    # import pdb
    # pdb.set_trace()

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    for steps in range(attack_args["attack_iterations"]):
        def closure():
            optimizer.zero_grad()
            delta1, delta2 = transform_delta(
                nw_delta1, nw_delta2, attack_args["attack_epsilon"], homogeneous=homogeneous, pcfa_deltaconstraint=attack_args["pcfa_deltaconstraint"]
            )
            nw_input1_closure, nw_input2_closure = image1 + delta1, image2 + delta2

            # Predict the flow, InputModel does clipping to valid image range
            flow_closure = model(inputs, nw_input1_closure, nw_input2_closure)[
                "flows"
            ].squeeze(0)
            flow_closure = flow_closure.to(device)

            loss_closure = get_loss(
                f_type=attack_args["attack_loss"],
                pred=flow_closure,
                target=target
            )
            loss_closure.backward()

            return loss_closure

        # Update the optimization parameters
        optimizer.step(closure)

        delta1, delta2 = transform_delta(
            nw_delta1, nw_delta2, attack_args["attack_epsilon"], homogeneous=homogeneous, pcfa_deltaconstraint=attack_args["pcfa_deltaconstraint"]
        )
        nw_input1, nw_input2 = image1 + delta1, image2 + delta2
        
        if attack_args["pcfa_boxconstraint"] != "change_of_variables":
            # clamp delta to image range
            nw_input1 = nw_input1.clamp(0.0, 1.0)
            nw_input2 = nw_input2.clamp(0.0, 1.0)
            
            delta1 = nw_input1 - image1
            delta2 = nw_input2 - image2

        # Re-predict flow with the perturbed image, and update the flow prediction for the next iteration
        with torch.no_grad():
            preds = model(inputs, nw_input1, nw_input2)
            flow_pred = preds["flows"].squeeze(0)
            flow_pred = flow_pred.to(device)

            iteration_metrics = iteration_metrics | losses.calc_delta_metrics(
                delta1, delta2, steps + 1
            )
            iteration_metrics = iteration_metrics | losses.calc_epe_metrics(
                model.model_loaded, preds, inputs, steps + 1, targeted_inputs
            )

    return (
        preds,
        delta1, delta2,
        iteration_metrics,
    )


def transform_delta(
    nw_delta1: torch.Tensor, nw_delta2: torch.Tensor, delta_bound: float, homogeneous: torch.Tensor, pcfa_deltaconstraint: Literal["project", "homogeneous_scale"]
):
    """Enforce delta bound on delta12"""
    n = torch.numel(nw_delta1) + torch.numel(nw_delta2)
    if pcfa_deltaconstraint == "homogeneous_scale":
        two_norm_squared = (
            nw_delta1.pow(2).sum() + nw_delta2.pow(2).sum() + homogeneous.abs()
        )
        scale = (
            delta_bound * torch.sqrt(n / two_norm_squared)
        )
    else:
        two_norm_squared = nw_delta1.pow(2).sum() + nw_delta2.pow(2).sum()
        scale = torch.clamp(delta_bound * torch.sqrt(n / (two_norm_squared + 1e-12)), max=1.0)
        
    delta1 = nw_delta1 * scale
    delta2 = nw_delta2 * scale
    
    return delta1, delta2
