from jsonargparse import Namespace
import json

from pathlib import Path
from ptlflow import get_model_reference
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.registry import RegisteredModel
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from loguru import logger
from typing import Dict, Optional, Tuple

from .attacks import _init_parser


def get_default_cli(model_name: str, dataset: str) -> PTLFlowCLI:
    """Return default arguments and data loader. To be used for the Python API."""
    # merge threat arguments with optical flow model-specific arguments
    parser = _init_parser()
    manual_args = Namespace()

    manual_args.model = model_name
    manual_args['data.val_dataset'] = dataset

    manual_args.ckpt_path = get_pretrained_ckpt(dataset)
    logger.info(f'Set ckpt_path to {manual_args.ckpt_path}')

    manual_args.write_outputs = True
    
    logger.info(f'Manual Arguments: {manual_args}')

    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=FlowDataModule,
        parser_kwargs={'parents': [parser]},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
        args=manual_args
    )

    return cli


def get_pretrained_ckpt(dataset: str) -> str:
    pretrained_ckpts = {
        'kitti-2015': 'kitti',
        'sintel-clean': 'sintel',
        'sintel-final': 'sintel'
    }
    return pretrained_ckpts.get(dataset, dataset)


def get_results(result_path: Path) -> Dict[str, Optional[float]]:
    if not result_path.is_file():
        raise FileNotFoundError(f'JSON file not found: {result_path}.')
    
    with open(result_path,'r') as f:
        data = json.load(f)
    
    metrics = ['epe', 'px3', 'cosim']
    results = {metric: data['experiments'][-1]['metrics'].get(metric) for metric in metrics}
    return results
