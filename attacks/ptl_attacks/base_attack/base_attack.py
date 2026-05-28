
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from attacks.data.flow_datamodule_extended import FlowDataModuleFlowBench

class BaseAttack(object):

    def __init__(self,
                 precomputed: bool = False,
                 targeted: bool = False,
                 target: Optional[str] = None):
        super().__init__()
        self.precomputed = precomputed
        self.targeted = targeted
        self.target = target
        
        self.attack_targeted = targeted # for compability with old parameter naming
        self.attack_target = target # for compability with old parameter naming
        if self.targeted and self.target is None:
            raise ValueError("You must pass a target for the targeted attack. Choose from 'negative' and 'zero'.")
    
    def make_target(self, inputs, preds) -> torch.Tensor:
        assert self.targeted, "This attack is not targeted. You can only make targets for a targeted attack."

        if self.target == "negative":
            targeted_flow_tensor = -preds["flows"]
        elif self.target == "zero":
            targeted_flow_tensor = torch.zeros_like(preds["flows"])
        else:
            raise ValueError(f"Invalid target: {self.target}. Choose from 'negative' and 'zero'.")
        return targeted_flow_tensor

    def __call__(self, *args, **kwargs):
        """Calls self.forward(*args, **kwargs)"""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Parameters
        ----------
        inputs: Dict[str, Any]
            For compatibility with the framework, the input should be a dict containing at least the following keys:

            - 'model_inputs': Dict[str, torch.Tensor], the unperturbed inputs to the model

            - 'model': BaseModel, the model to be attacked

            - 'targeted_inputs': Dict[str, torch.Tensor], optional, only used if the attack is targeted.
              A Dict similar to 'model_inputs' but with the 'flow' entry replaced with the target flow.

        Returns
        -------
        Dict[str, torch.Tensor]
            For compatibility with the framework, the output should be a dict containing at least the following keys:

            - 'perturbed_inputs': a dict matching the inputs format of ptlflow models.
              In particular, it should have the key "images" containing the perturbed images.

            - 'preds': a dict matching the output format of ptlflow models
              In particular, it should have the key "flows" containing the predicted flow of the input model on  perturbed images.
              
            - 'iteration_metrics' (optional): additional metrics to be logged

        """
        raise NotImplementedError()

    def preprocess_dataloader(self, dataset_name: str, data_module: FlowDataModuleFlowBench) -> DataLoader:
        """
        Get the dataloader to be used for the attack if precomputed is True.
        Sub-classes may overwrite this method for attack-specific logic.
        """
        return data_module.val_dataloaders()[0]
