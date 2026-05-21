
from typing import Any, Dict, Optional
from lightning import LightningModule
import torch
from torch.utils.data import DataLoader

from attacks.data.flow_datamodule_extended import FlowDataModuleFlowBench

class BaseAttack(LightningModule):

    def __init__(self,
                 precomputed: bool = False,
                 targeted: bool = False,
                 target: Optional[str] = None):
        super().__init__()
        self.precomputed = precomputed
        self.targeted = targeted
        self.target = target
        
        self.save_hyperparameters()
    
    def make_target(self, inputs, preds) -> torch.Tensor:
        if self.target == "negative":
            targeted_flow_tensor = -preds["flows"]
        else:
            targeted_flow_tensor = torch.zeros_like(preds["flows"])
        return targeted_flow_tensor

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Parameters
        ----------
        inputs: Dict[str, Any]
            For compatibility with the framework, the input should be a dict containing at least the following keys:

            - 'model_inputs': Dict[str, torch.Tensor], the unperturbed inputs to the model

            - 'model': BaseModel, the model to be attacked

            - 'targetted_inputs': Dict[str, torch.Tensor], optional, only used if the attack is targeted.
              A Dict similar to 'model_inputs' but with the 'flow' entry replaced with the target flow.

        Returns
        -------
        Dict[str, torch.Tensor]
            For compatibility with the framework, the output should be a dict containing at least the following keys:

            - 'flows': a 5D tensor BNCHW containing the predicted flows. The flows must be at the original scale, i.e.,
              their size is the same as the input images, and the flow magnitudes are scaled accordingly. Most networks only
              produce a single 2D optical flow per batch element, so the output shape will be B12HW. N can be larger than one
              if the network produces predictions for a larger temporal window.

            - 'occs': optional, and only included if the network also predicts occlusion masks. It is a 5D tensor following the
              same structure as 'flows'.

            - 'mbs': same as 'occs' but for occlusion masks.

            - 'confs': same as 'occs' but for flow confidence predictions.

            The keys above are used by other parts of PTLFlow (if available). The output can also include any other keys as
            well. For example, by default, the output of forward() will be passed to the loss function. So the output may
            include keys which are going to be used for computing the training loss.
        """
        raise NotImplementedError()

    def preprocess_dataloader(self, dataset_name: str, data_module: FlowDataModuleFlowBench) -> DataLoader:
        
        return data_module.val_dataloaders()[0]
