
""" Does the same as flowbench/attacks.py but may be modified to register external models to ptlflow.
Arguments can be given as config.yaml or command line arguments.
 
Sample usage:

    python -m sample_usage.do_attack \
        --model my_very_own_raft \
        --ckpt_path path/to/checkpoint.pth \
        --attack pcfa \
        --data.val_dataset sintel-final+kitti-2015 \
        --output_path outputs/validate
"""

from pathlib import Path

from jsonargparse import Namespace
from loguru import logger
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel
from ptlflow import restore_model

from attacks.data.flow_datamodule_extended import FlowDataModuleFlowBench
from flowbench.attacks import attack, _init_parser

from .my_new_model import * # registers model in ptlflow, so the passed model doesn't need to be fully qualified

if __name__ == "__main__":
    parser = _init_parser() # Parser for attack arguments
    
    # The PTLFlowCLI instantiates the model and data module.
    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=FlowDataModuleFlowBench,
        parser_kwargs={"parents": [parser]}, # combine PTLFlow's CLI with our attack argument parser
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
        # # You could pass arguments here.
        # # (This is useful in case you want a self-contained Python program without command line arguments and configs.)
        # # However, then you can not pass program arguments.
        # args=Namespace(
        #     model=Namespace(class_path="my_new_model.raft.my_very_own_raft"),
        #     ckpt_path="sintel",
        #     data=Namespace(val_dataset="sintel-final", dataset_config_path="../datasets.yaml"),
        #     attack=["pcfa"]
        # )
    )

    cfg = cli.config
    logger.info(f"{cfg}")
    
    # Change output path based on other parameters. Modify the following lines of code to your preferred output structure.
    # Cf. flowbench/evals.py for FlowBench's standardized folder structure. (I'm not copying that here because it would be too much code!)
    cfg.model_name = cfg.model.class_path.split(".")[-1]
    model_id = cfg.model_name
    if cfg.ckpt_path is not None:
        model_id += f"_{Path(cfg.ckpt_path).stem}"
    cfg.output_path = Path(cfg.output_path) / model_id
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

    logger.info("The outputs will be saved to {}.", cfg.output_path)

    # Load checkpoint if specified.
    model = cli.model
    model = restore_model(model, cfg.ckpt_path)
    
    metrics_df = attack(cfg, model, cli.datamodule)
