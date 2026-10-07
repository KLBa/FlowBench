# FlowBench: A Robustness Benchmark for Optical Flow Estimation

This repository provides comprehensive tools and pre-trained models for benchmarking the robustness of optical flow estimation algorithms.

> This repository is a fork of [shashankskagnihotri/FlowBench](https://github.com/shashankskagnihotri/FlowBench).
>
> [Link to Paper](https://openreview.net/forum?id=Kh4bj6YDNm)
>
> [YouTube Video](https://youtu.be/_AUKi_48vmc)
>
> Accepted at Transactions on Machine Learning Research (TMLR), 2025

---

## Installation

This package requires **Python 3.1x.x** (tested with 3.14.3). Please ensure you have a compatible Python version installed.

### Step 1: Clone the repository including submodules

```bash
git clone --recurse-submodules <url>
# If you cloned without --recurse-submodules, run: 
git submodule update --init --recursive
```

### Step 2: Install dependencies

FlowBench requires [`ptlflow`](https://github.com/hmorimitsu/ptlflow) for loading models.
I recommend a local install

```bash
git clone git@github.com:hmorimitsu/ptlflow.git
cd ptlflow
pip install -e .
cd ..
```

Other dependencies:

```bash
pip install "setuptools<81" imagecorruptions cospgd
```

### Step 3: Install this package in development mode

```bash
pip install -e .
```

### Step 4: Test run

```bash
python -m flowbench.attacks_refactored --model raft --ckpt_path chairs --data.val_dataset kitti-2015-trainval --attack_obj pcfa --attack_obj.attack_iterations 5
```

Install any dependencies as they occur... Sorry for not keeping this consistent.

---

## Datasets

### KITTI2015

1. Download the KITTI 2015 dataset from the [KITTI Scene Flow Benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
2. After extraction, verify that the contents include `training/` and `testing/` directories.
3. Place the dataset at `datasets/kitti2015` or adjust the path in `ptlflow/datasets.yaml`.

### MPI Sintel

1. Download the MPI Sintel dataset from the [MPI Sintel Flow Dataset](http://sintel.is.tue.mpg.de/downloads).
2. After extraction, verify that the contents include `training/` and `test/` directories.
3. Place the dataset at `datasets/Sintel`.
4. Download the MPI Sintel Depth training data from the [MPI Sintel Depth Training Data](http://sintel.is.tue.mpg.de/depth).
5. Extract the archive and verify it contains `training/camdata_left`, `training/depth`, and `training/depth_viz`. Place these directories under `datasets/Sintel/training` or adjust the path in `ptlflow/datasets.yaml`.

Alternatively, use our convenience script:

```bash
bash download_mpi_sintel.sh
```

### 3D Common Corruptions Images

Download the precomputed 3D Common Corruption Images for KITTI2015 and MPI Sintel using the script below:

```bash
bash download_3dcc_data.sh
```

After download, the directory structure should look like:

```
datasets/3D_Common_Corruption_Images/kitti2015
datasets/3D_Common_Corruption_Images/Sintel
```

**Note:** Even if you only want to evaluate on the 3D Common Corruption Images, you must still download the original datasets to the expected folders to avoid errors.

### Adversarial Weather

There are two options:

1) Download precomputed weather particle files via script (or manually from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3677):

```bash
bash download_weather_files.sh
```

2) Generate your own variants with custom parameters. Follow the instructions from the [DistractingDownpour repository](https://github.com/cv-stuttgart/DistractingDownpour).

---

## How to Use

### Python API

> :warning: I'm not using the Python API and `flowbench.evals`, so they're likely to be broken on this fork.
> Skip to the [Commandline Interface](#command-line-interface).

<details><summary>Click to view old Python interface.</summary>

#### Model Zoo

```python
from flowbench.evals import load_model

model = load_model(
    model_name='RAFT',
    dataset='KITTI2015',
)
```

#### Supported Models

To browse the full list of supported models:

- See [`SUPPORTED_MODELS.md`](./SUPPORTED_MODELS.md)

#### Evaluation

##### Adversarial Attacks

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='KITTI2015',
    retrieve_existing=True,
    threat_model='PGD',
    iterations=20, epsilon=8/255, alpha=0.01,
    lp_norm='Linf', optim_wrt='ground_truth',
    targeted=True, target='zero',
)
```

- `retrieve_existing`: When `True` and a matching evaluation exists in the benchmark, returns the cached result. Otherwise, runs a new evaluation.
- `threat_model`: The type of adversarial attack
- `iterations`: Number of attack iterations
- `epsilon`: Permissible perturbation budget (ε)
- `alpha`: Step size of the attack (ϑ)
- `lp_norm`: Norm used to bound perturbation. Supported values: `'Linf'` or `'L2'`
- `targeted`: Boolean flag indicating whether the attack is targeted
- `target`: Target flow for a targeted attack (only applicable if `targeted=True`). Supported values: `'zero'` or `'negative'`
- `optim_wrt`: Flow used as a reference for optimization. Supported values: `'ground_truth'` or `'initial_flow'`

##### Adversarial Weather

```python
# demo.py
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='Sintel-Final',
    retrieve_existing=False,
    threat_model='Adversarial_Weather',
    weather='snow',
    num_particles=10000,
    targeted=True,
    target='zero',
    weather_data="datasets/adv_weather_data/weather_particles_red",
)
```

See the docstring of `evaluate()` for additional configuration options.

To use evaluate via command line:

```bash
python demo.py --weather_data path_to_particle_data
```

If you used `download_weather_files.sh`, three variants of particle data for the Sintel dataset should be available under `datasets/adv_weather_data/`. For example:

```bash
python demo.py --weather_data datasets/adv_weather_data/weather_snow_3000
```

- `retrieve_existing`: Works as described above
- `threat_model`: Must be `'Adversarial_Weather'`
- `weather`: Weather condition for the adversarial weather attack. Supported values: `'snow'`, `'fog'`, `'rain'`, or `'sparks'`
- `num_particles`: Number of particles per frame
- `targeted`: Boolean flag indicating whether the attack is targeted
- `target`: Target flow for a targeted attack (only applicable if `targeted=True`). Supported values: `'zero'` or `'negative'`

##### 2D Common Corruptions

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='KITTI2015',
    retrieve_existing=True,
    threat_model='2DCommonCorruption',
    severity=3,
)
```

- `retrieve_existing`: Works as described above
- `threat_model`: Must be `'2DCommonCorruption'`; returns evaluations across 15 corruption types
- `severity`: An integer from 1 to 5 indicating the corruption severity

##### 3D Common Corruptions

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='KITTI2015',
    retrieve_existing=True,
    threat_model='3DCommonCorruption',
    severity=3,
)
```

- `retrieve_existing`: Works as described above
- `threat_model`: Must be `'3DCommonCorruption'`; returns evaluations across 8 corruption types
- `severity`: An integer from 1 to 5 indicating the corruption severity
</details>

### Command Line Interface

FlowBench offers a command line interface inspired by ptlflow's scripts.

Example:

```sh
python -m flowbench.attacks_refactored \
    --model raft --ckpt_path chairs --model.iters 1 \
    --data.val_dataset kitti-2015-trainval --data.dataset_config_path datasets.yaml \
    --attack_obj pcfa --attack_obj.attack_iterations 1 --attack_obj.attack_epsilon 0.05 --attack_obj.attack_alpha 1e-7
```

FlowBench's CLI extends ptlflow's CLI.
The most important arguments include:

#### ptlflow-inherited arguments

All arguments defining the model and data are inherited from ptlflow.

- `--model` the name of the model or a dot path to its definition, use `--model.property_name` to change a hyperparameter of the model
- `--ckpt_path` a file path to the checkpoint or a shorthand (typically `chairs`, `things`, `sintel`) defined by ptlflow
- `--data.val_dataset` the dataset the attack will be performed on
  - (Note that `--data.train_dataset` and `--data.test_dataset` are ignored!)
- `--data.dataset_config_path` a path to a yaml file defining the dataset paths

#### FlowBench-specific args

FlowBench adds arguments defining the attack and logging.

Attack args:

- `--attack_obj` an attack name or a dot path to its definition
  - To pass attack-specific arguments, e.g., number of iterations and the perturbation limit $\varepsilon$, use the dot notation from jsonargparse, e.g.
  
  ```sh
  --attack_obj.attack_iterations 3 --attack_obj.attack_epsilon 0.001
  ```

Logging args:

- `--output_path /path/to/outputs` path to write outputs to. `flowbench.attacks_refactored` always writes a JSON containing the experiment results.
- `--write_outputs` set this flag to store the perturbed images to the output path (Note that this may explode your storage.)

### Metric Names

FlowBench logs experiments under `args.output_path`, on this fork the default is outputs/flowbench (main defaults to outputs/validate iirc)

> :warning: Metrics with `own_` pre-prepended ignore the `valid` map.
> This is fine when comparing to anything except ground truth...  

This fork renamed metrics:

| Previous name                     | New name  | Meaning   |
|---                                |---        |---        |
| epe                               | epe_attacked_to_target          | _If targeted:_ AEPE between attack target and attacked flow. |
| epe                               | epe_attacked_to_groundtruth          | _If not targeted:_ epe between ground truth and attacked flow. |
| px1                               | px1_attacked_to_target          | _If targeted:_ px1 between attack target and attacked flow. |
| px1                               | px1_attacked_to_groundtruth          | _If not targeted:_ px1 between ground truth and attacked flow. |
| px3                               | px3_attacked_to_target          | _If targeted:_ px3 between attack target and attacked flow. |
| px3                               | px3_attacked_to_groundtruth          | _If not targeted:_ px3 between ground truth and attacked flow. |
| px5                               | px5_attacked_to_target          | _If targeted:_ px5 between attack target and attacked flow. |
| px5                               | px5_attacked_to_groundtruth          | _If not targeted:_ px5 between ground truth and attacked flow. |
| flall                             | flall_attacked_to_target          | _If targeted:_ flall between attack target and attacked flow. |
| flall                             | flall_attacked_to_groundtruth          | _If not targeted:_ flall between ground truth and attacked flow. |
| wauc                              | wauc_attacked_to_target          | _If targeted:_ wauc between attack target and attacked flow. |
| wauc                              | wauc_attacked_to_groundtruth          | _If not targeted:_ wauc between ground truth and attacked flow. |
| own_epe                           | own_epe_attacked_to_target          | _If targeted:_ AEPE between attack target and attacked flow. |
| own_epe                           | own_epe_attacked_to_groundtruth          | _If not targeted:_ AEPE between ground truth and attacked flow. |
| epe_orig_preds                    |           | AEPE between unattacked flow and attacked flow |
| own_epe_orig_preds                |           | (same as epe_orig_preds but differently calculated) |
| cosim_target                      |           | Cosine similarity between attack target and attacked flow |
| cosim_orig_preds                  |           | Cosine similarity between unattacked flow and attacked flow |
| epe_ground_truth                  |           | AEPE between ground truth and attacked flow |
| own_epe_ground_truth              |           | (same as epe_ground_truth but differently calculated) |
| cosim_ground_truth                |           | Cosine similarity between ground truth and attacked flow          |
| epe_initial_to_negative           |           | AEPE between attacked flow and negative unattacked flow |
| epe_initial_to_zero               |           | AEPE between attacked flow and zero flow (that is average flow magnitude) |
| epe_ground_truth_to_negative      |           | AEPE between ground truth and negative unattacked flow          |
| epe_ground_truth_to_zero          |           | AEPE between ground truth and zero flow (that is average flow magnitude) |
| own_epe_ground_truth_to_negative  |           | (same as epe_ground_truth_to_negative but differently calculated) |
| own_epe_ground_truth_to_zero      |           | (same as own_epe_ground_truth_to_zero but differently calculated) |
| l2_delta1                         |           | l2 norm of perturbation on frame 1           |
| l2_delta2                         |           | l2 norm of perturbation on frame 2           |
| l2_delta12                        |           | l2 norm of combined perturbation on both frames           |
| l0_delta1                         |           | l0 norm of perturbation on frame 1           |
| l0_delta2                         |           | l0 norm of perturbation on frame 2           |
| l0_delta12                        |           | l0 norm of combined perturbation on both frames           |
| l_inf_delta1                      |           | l_inf norm of perturbation on frame 1         |
| l_inf_delta2                      |           | l_inf norm of perturbation on frame 2         |
| l_inf_delta12                     |           | l_inf norm of combined perturbation on both frames           |

Abbreviations:

AEPE = Average endpoint error

## CITATION

If you use this work, please cite:

```bibtex
@article{
agnihotri2025flowbench,
title={FlowBench: Benchmarking Optical Flow Estimation Methods for Reliability and Generalization},
author={Shashank Agnihotri and Julian Yuya Caspary and Luca Schwarz and Xinyan Gao and Jenny Schmalfuss and Andres Bruhn and Margret Keuper},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=Kh4bj6YDNm},
note={}
}
```

or

```txt
Agnihotri, Shashank, Julian Yuya Caspary, Luca Schwarz, Xinyan Gao, Jenny Schmalfuss, Andrés Bruhn, and Margret Keuper. “FlowBench: Benchmarking Optical Flow Estimation Methods for Reliability and Generalization.” Transactions on Machine Learning Research, 2025. https://openreview.net/forum?id=Kh4bj6YDNm
```
