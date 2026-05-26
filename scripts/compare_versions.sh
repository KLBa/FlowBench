#!/bin/bash
#SBATCH --job-name="flowbench_compare_versions"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=CompareVersions
#SBATCH --output=slurm/CompareVersions.out
#SBATCH --gres=gpu:1
#SBATCH --error=slurm/CompareVersions.out

source /data/katrin/venvs/venv-sari/bin/activate

python debugging/compare_versions.py
