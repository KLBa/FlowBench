# Sample usage

This folder contains some (hopefully) independent scripts on how to run an attack from FlowBench on a custom model.

## Folder Contents

* [my_new_model/](my_new_model/) defines a custom model for the demo.

  (Here we just extend the RAFT model defined by ptlflow and register it under a new name. Cf. [the ptlflow docs](https://ptlflow.readthedocs.io/en/latest/custom/new_model.html) for how to define a model that is compatible with PTLFlow.)

* [attack_pcfa_cmd.sh](attack_pcfa_cmd.sh) shows how to run an attack using `flowbench.attack` and command line arguments only
* [attack_pcfa_config.sh](attack_pcfa_config.sh) shows how to run an attack using `flowbench.attack` and a config file

  (Command line arguments and config files are interchangeable. Praised be Pytorch Lightning!)
* [do_attack.py](do_attack.py) does basically the same as `flowbench.attack` but may serve as reference Python implementation to be extended
* **not recommended until I understand `flowbench.evals` better:**
  
  [do_evaluate.py](do_evaluate.py) shows how to call the `flowbench.evals.evaluate()` API.
  That interface is a bit more restrictive:
  * Registering the model beforehand is required.
  * Some default arguments cannot be changed which leads to following issues:
    * You can't set the `data.dataset_config_path`. The working directory must contain the `datasets.yaml`.
    * `output_path` is fixed
    * Only specific attack parameters can be passed (I'm not sure how extensive that is.)

  Nonetheless, using `evaluate()` has the advantage that the folder structure is standardized and the output path need not be re-invented.

  Cf. main function of `flowbench.evals` for more sample calls.

## Setup

Scripts in this folder assume that `ptlflow` and `flowbench` are installed in your environment.
Check the [README](https://github.com/shashankskagnihotri/FlowBench/tree/flowbench#installation) of FlowBench for details.

Basically that boils down to the following installs (+ requirements of the modules which you can install as you encounter them).

```sh
cd ../_ptlflow # PTLFlow repo root
pip install -e .

cd .. # FlowBench repo root
pip install -e .
```

If your Python version is not supported (e.g., currently 3.14 works for me but pip doesn't know), try passing `--ignore-requires-python`.
