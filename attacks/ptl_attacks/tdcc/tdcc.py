
from typing import Any, Dict
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from ..base_attack.base_attack import BaseAttack

from ...data.flow_datamodule_extended import FlowDataModuleFlowBench


class CommonCorruptions3D(BaseAttack):

    def __init__(self,
                 tdcc_corruption: str,
                 tdcc_intensity: int,
                 precomputed: bool = True):
        super().__init__(precomputed=precomputed)
        self.tdcc_corruption = tdcc_corruption
        self.tdcc_intensity = tdcc_intensity
        
        if not self.precomputed:
            raise NotImplementedError()
        self.save_hyperparameters()

    def forward(self, inputs: Dict[str, Any]):
        if self.precomputed:
            inputs_model = inputs['inputs']
            model = inputs['model']
            perturbed_inputs = inputs_model["inputs_perturbed"]
            assert inputs_model["images"].device == perturbed_inputs["images"].device, (inputs_model["images"].device, perturbed_inputs["images"].device)
            preds = model(perturbed_inputs)
            return {
                "preds": preds,
                "perturbed_inputs": perturbed_inputs
            }
        else:
            raise NotImplementedError()

    def preprocess_dataloader(self, dataset_name: str, data_module: FlowDataModuleFlowBench):
        
        logger.warning(f'Dataloader {dataset_name}')
        vals = dataset_name.split('-')
        vals[0] = f"{vals[0]}_precomputed"
        vals.append(f"tdcc_{self.tdcc_intensity}_{self.tdcc_corruption}")
        vals.append(f"flowbench_attach")
        data_module.val_dataset = "-".join(vals)
        data_module.val_dataset_parsed = None
        data_module.setup("validate")
        return data_module.val_dataloader()[0]
