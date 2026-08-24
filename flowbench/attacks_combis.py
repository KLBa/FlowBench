"""
Runs flowbench.attack multiple times on a combination of configs.
That is, you can pass up to three lists of configs.

python -m flowbench.attacks_combis --config configs/combis/raft-pcfa.yaml --data.dataset_config_path ../ptlflow-mirror/datasets_sarinagara.yaml --ignore_errors
"""

from glob import glob
import itertools
import os
from pathlib import Path
import sys
from typing import List
from jsonargparse import ArgumentParser
from loguru import logger

from ptlflow import restore_model
from ptlflow.utils.registry import RegisteredModel
from .attacks import FlowDataModuleFlowBench, PTLFlowCLI, attack
from .attacks import _init_parser as _init_attacks_parser


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config_1",
        type=List[str],
        default=[],
        help=("List of configs."),
    )
    
    parser.add_argument(
        "--config_2",
        type=List[str],
        default=[],
        help=("List of configs."),
    )
    
    parser.add_argument(
        "--config_3",
        type=List[str],
        default=[],
        help=("List of configs."),
    )
    parser.add_argument(
        "--ignore_errors",
        action="store_true",
        help=("When an evaluation of one config combination fails, continue with the next."),
    )
    
    return parser


def preprocess_config_list(config_paths):
    res = []
    for c in config_paths:
        expanded = glob(c)
        res.extend(expanded)
        if len(expanded) == 0:
            raise FileNotFoundError(f"Config path '{c}' does not exist.")
    return res


if __name__ == "__main__":
    parser_configs = _init_parser()
    parser_attacks = _init_attacks_parser()
    
    args = sys.argv[1:] + ["--model", "raft"]
    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=FlowDataModuleFlowBench,
        parser_kwargs={'parents': [parser_configs, parser_attacks]},
        run=False,
        args=args,
        parse_only=False,
        auto_configure_optimizers=False,
    )
    cfg = cli.config
    
    ignore_errors = cli.config["ignore_errors"]
    config_lists = [
        cli.config["config_1"],
        cli.config["config_2"],
        cli.config["config_3"]
    ]
    # expand globs and remove empty lists
    config_lists = [preprocess_config_list(x) for x in config_lists if len(x) > 0]
    
    combis = itertools.product(*config_lists)
    for config_paths in combis:
        try:
            args = sys.argv[1:]
            for c in config_paths:
                args.extend(["--config", c])

            cli = PTLFlowCLI(
                model_class=RegisteredModel,
                subclass_mode_model=True,
                datamodule_class=FlowDataModuleFlowBench,
                parser_kwargs={'parents': [parser_configs, parser_attacks]},
                run=False,
                args=args,
                parse_only=False,
                auto_configure_optimizers=False,
            )
            cfg = cli.config
            
        
            cfg.model_name = cfg.model.class_path.split(".")[-1]
            model_id = cfg.model_name

            if cfg.ckpt_path is not None:
                model_id += f"_{Path(cfg.ckpt_path).stem}"
            if cfg.max_forward_side is not None:
                model_id += f"_maxside{cfg.max_forward_side}"
            if cfg.scale_factor is not None:
                model_id += f"_scale{cfg.scale_factor}"
            if cfg.model.init_args.warm_start:
                model_id += f"_warm"
            cfg.output_path = Path(cfg.output_path) / model_id
            Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

            logger.info("The outputs will be saved to {}.", cfg.output_path)

            model = cli.model
            model = restore_model(model, cfg.ckpt_path)
        
            metrics_df = attack(cfg, model, cli.datamodule)
            
            for fn in glob(os.path.expanduser("~/.cache/torch/hub/checkpoints/*.ckpt")):
                os.remove(fn)
        except Exception as e:
            logger.error(f"Config combination {config_paths} failed with error:")
            logger.error(f"{type(e)}: {e}")
            if not ignore_errors:
                raise e
