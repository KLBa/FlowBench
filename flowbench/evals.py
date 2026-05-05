import os
from jsonargparse import Namespace
import pandas as pd
import math

from ptlflow import get_model, get_model_reference
from ptlflow.utils.utils import get_list_of_available_models_list
from pathlib import Path

from .attacks import attack
from .utils import get_default_cli, get_pretrained_ckpt, get_results


common_corruptions_2d = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

common_corruptions_3d = [
    'far_focus', 'near_focus', 'fog_3d', 'color_quant',
    'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur',
]

attacks = ['fgsm', 'bim', 'pgd', 'cospgd', 'apgd', 'fab', 'pcfa']


def load_model(model_name: str, dataset: str):
    """
    Load a pre-configured optical flow estimation model with pretrained weights.

    Args:
        model_name (str): Name of the optical flow model (e.g., CCMR, RAFT).
        dataset (str): Name of the dataset the model was trained on (e.g., KITTI-2015).
    """
    model_name = model_name.lower()
    if model_name not in get_list_of_available_models_list():
        raise ValueError(f"Model '{model_name}' is not recognized.")

    dataset = dataset.lower()
    if dataset == 'kitti2015':
        dataset = 'kitti-2015'
    pretrained_ckpt = get_pretrained_ckpt(dataset)

    model_ref = get_model_reference(model_name)
    checkpoints = model_ref.pretrained_checkpoints.keys()

    if pretrained_ckpt in checkpoints:
        model = get_model(model_name, pretrained_ckpt)
    else:
        raise ValueError(f"No pretrained weights for model '{model_name}' on dataset '{pretrained_ckpt}'.")
    return model


def evaluate(
    model_name: str, dataset: str,
    retrieve_existing: bool = True, threat_model: str = None,
    iterations: int = 20, epsilon: float = 8/255, alpha: float = 0.01,
    lp_norm: str = 'Linf', targeted: bool = True, target: str = 'zero', optim_wrt: str = 'ground_truth',
    weather: str = 'snow', num_particles: int = 10000,
    severity: int = 3,
    model_args: Namespace | None = None,
):
    """
    Evaluate the robustness of optical flow estimation under various conditions, including
    adversarial attacks, adversarial weather, 2D common corruptions, and 3D common corruptions.

    This function provides a simplified interface to `flowbench.attacks.attack(...)`, e.g. sets args.output_path to a standardized path,
    as well as automatic retrieving existing results from eval_csv/evals.csv without re-running experiments.

    Args:
        model_name (str): Name of the optical flow model (e.g., 'CCMR', 'RAFT').
        dataset (str): Dataset the model was trained on (e.g., 'KITTI-2015').
        retrieve_existing (bool): If True, returns cached results if available; otherwise, reruns the evaluation.
        threat_model (str): Type of perturbation to apply. None indicates clean/unperturbed images.
        iterations (int): Number of attack iterations (used in adversarial attack and adversarial weather).
        epsilon (float): Perturbation budget ε (used in adversarial attack).
        alpha (float): Attack step size ϑ (used in adversarial attack).
        lp_norm (str): Norm used to constrain perturbations (used in adversarial attack).
        targeted (bool): Whether the attack is targeted (used in adversarial attack and adversarial weather).
        target (str): Target flow for a targeted attack (used in adversarial attack and adversarial weather).
        optim_wrt (str): Flow reference for optimization (used in adversarial attack).
        weather (str): Weather type for adversarial weather attack(used in adversarial weather).
        num_particles (int): Number of particles per frame (used in adversarial weather).
        severity (int): Corruption severity level (1–5) used in 2D and 3D common corruptions.
        attack_loss (str): Set the name of the used loss function (mse, epe, cosim).
        model_args (Namespace):
            Parameters for the model.
            (e.g., for RAFT, if model_args=Namespace(iters=12), we initialize RAFT(iters=12))
    """
    model_name = model_name.lower()
    if model_name not in get_list_of_available_models_list():
        raise ValueError(f"Model '{model_name}' is not recognized.")

    dataset = dataset.lower()
    if dataset == 'kitti2015':
        dataset = 'kitti-2015'

    model = load_model(model_name, dataset)
    results = {}

    if threat_model:
        threat_model = threat_model.lower()

    if retrieve_existing:
        csv_path = 'eval_csv/evals.csv'
        df = pd.read_csv(csv_path)

    if threat_model is None:
        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == 'none')
            ]
            if not filtered.empty:
                metrics = ['epe', 'px3', 'cosim']
                results = {metric: float(filtered.iloc[0][metric]) for metric in metrics}
                return model, results

        # run on unperturbed and unaltered images
        ptlf_cli = get_default_cli(model_name, dataset)
        args = ptlf_cli.config
        data_module = ptlf_cli.datamodule
        args.output_path = Path(f'outputs/validate/{model_name}_{args.ckpt_path}/unperturbed')
        args.output_path.mkdir(parents=True, exist_ok=True)

        model = get_model(model_name, args.ckpt_path, model_args)
        attack(args, model, data_module)

        # load results from json
        results = get_results(args.output_path / f'metrics_{dataset}.json')
        return model, results

    elif threat_model in attacks:
        if lp_norm not in ['Linf', 'L2']:
            raise ValueError(f"Invalid lp_norm: {lp_norm}. Must be 'Linf' or 'L2'.")
        if targeted and target not in ['negative', 'zero']:
            raise ValueError(f"Invalid target: {target}. Must be 'negative' or 'zero' when targeted=True.")
        if optim_wrt not in ['ground_truth', 'initial_flow']:
            raise ValueError(f"Invalid optim_wrt: {optim_wrt}. Must be 'ground_truth' or 'initial_flow'.")

        lp_norm = 'inf' if lp_norm == 'Linf' else 'two'

        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == threat_model) &
                (df['norm'] == lp_norm) &
                (df['epsilon'].round(4) == round(epsilon, 4)) &
                (df['iterations'] == iterations) &
                (df['alpha'] == alpha) &
                (df['targeted'] == targeted) &
                (df['optim'] == optim_wrt)
            ]
            if not filtered.empty:
                if targeted:
                    filtered = filtered[filtered['target'] == target]

                metrics = ['epe', 'px3', 'cosim']
                results = {metric: float(filtered.iloc[0][metric]) for metric in metrics}
                return model, results

        # run Adversarial Attacks
        ptlf_cli = get_default_cli(model_name, dataset)
        args = ptlf_cli.config
        data_module = ptlf_cli.datamodule
        args.output_path = Path(f'outputs/validate/{model_name}_{args.ckpt_path}/{threat_model}')
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.attack = threat_model
        args.attack_optim_target = optim_wrt
        args.attack_norm = lp_norm
        args.attack_epsilon = epsilon
        args.attack_iterations = iterations
        args.attack_alpha = alpha
        args.attack_targeted = targeted
        args.attack_target = target

        model = get_model(model_name, args.ckpt_path, model_args)
        attack(args, model, data_module)

        # load results from json
        results = get_results(args.output_path / f'metrics_{dataset}.json')
        return model, results

    elif threat_model == 'adversarial_weather':
        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == 'weather') &
                (df['weather_type'] == weather) &
                (df['targeted'] == targeted)
            ]
            if not filtered.empty:
                if targeted:
                    filtered = filtered[filtered['target'] == target]

                metrics = ['epe', 'px3', 'cosim']
                results = {metric: float(filtered.iloc[0][metric]) for metric in metrics}
                return model, results

        # run Adversarial Weather
        ptlf_cli = get_default_cli(model_name, dataset)
        args = ptlf_cli.config
        data_module = ptlf_cli.datamodule
        args.output_path = Path(f'outputs/validate/{model_name}_{args.ckpt_path}/weather/{weather}')
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.attack = 'weather'
        args.attack_targeted = targeted
        args.attack_target = target

        model = get_model(model_name, args.ckpt_path, model_args)
        attack(args, model, data_module)

        # load results from json
        results = get_results(args.output_path / f'metrics_{dataset}.json')
        return model, results

    elif threat_model == '2dcommoncorruption':
        if severity < 1 or severity > 5:
            raise ValueError("Severity must be an integer between 1 and 5.")

        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == 'common_corruptions') &
                (df['severity'] == severity)
            ]
            if not filtered.empty:
                for corruption in common_corruptions_2d:
                    metrics = ['epe', 'px3', 'cosim']
                    row = filtered[filtered['name'] == corruption]
                    metric_values = {metric: float(row.iloc[0][metric]) for metric in metrics}
                    results[corruption] = metric_values
                return model, results

        # run 2D Common Corruption
        for corruption in common_corruptions_2d:
            ptlf_cli = get_default_cli(model_name, dataset)
            args = ptlf_cli.config
            data_module = ptlf_cli.datamodule
            args.output_path = Path(f'outputs/validate/{model_name}_{args.ckpt_path}/2dcc/{corruption}')
            args.output_path.mkdir(parents=True, exist_ok=True)
            args.attack = 'common_corruptions'
            args.cc_name = corruption
            args.cc_severity = severity
            model = get_model(model_name, args.ckpt_path, model_args)
            attack(args, model, data_module)

            # load results from json
            metric_values = get_results(args.output_path / f'metrics_{dataset}.json')
            results[corruption] = metric_values
        return model, results

    elif threat_model == '3dcommoncorruption':
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be an integer between 1 and 5. Severity is {severity}")

        if retrieve_existing:
            filtered = df[
                (df['model'] == model_name) &
                (df['dataset'] == dataset) &
                (df['checkpoint'] == get_pretrained_ckpt(dataset)) &
                (df['attack'] == '3dcc') &
                (df['3dcc_intensity'] == severity)
            ]
            if not filtered.empty:
                for corruption in common_corruptions_3d:
                    metrics = ['epe', 'px3', 'cosim']
                    row = filtered[filtered['3dcc_corruption'] == corruption]
                    metric_values = {metric: float(row.iloc[0][metric]) for metric in metrics}
                    results[corruption] = metric_values
                return model, results

        # run 3D Common Corruption
        for corruption in common_corruptions_3d:
            ptlf_cli = get_default_cli(model_name, dataset)
            args = ptlf_cli.config
            data_module = ptlf_cli.datamodule
            args.output_path = Path(f'outputs/validate/{model_name}_{args.ckpt_path}/3dcc/{corruption}')
            args.output_path.mkdir(parents=True, exist_ok=True)
            args.attack = '3dcc'
            args.tdcc_corruption = corruption
            args.tdcc_intensity = severity

            model = get_model(model_name, args.ckpt_path, model_args)
            attack(args, model, data_module)

            # load results from json
            metric_values = get_results(args.output_path / f'metrics_{dataset}.json')
            results[corruption] = metric_values
        return model, results

    else:
        ValueError(f'Threat model {threat_model} is not supported')

    return model, results


if __name__ == "__main__":
    model = load_model(
        model_name='RAFT',
        dataset='KITTI2015',
    )

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=True,
        model_args=Namespace({'model.iters': 32})
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=False,
        threat_model='PGD',
        iterations=20, epsilon=8/255, alpha=0.01, lp_norm='Linf',
        targeted=True, target='zero', optim_wrt='ground_truth',
        model_args=Namespace({'model.iters': 32})
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='Sintel-Final',
        retrieve_existing=False,
        threat_model='Adversarial_Weather',
        weather='snow', num_particles=10000, targeted=True, target='zero',
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=False,
        threat_model='2DCommonCorruption',
        severity=3,
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=False,
        threat_model='3DCommonCorruption',
        severity=3,
    )
    print(results)
