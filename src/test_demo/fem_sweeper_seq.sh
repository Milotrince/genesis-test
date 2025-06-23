#!/bin/bash
#SBATCH --job-name=fem_sweeper
#SBATCH --output=logs/slurm_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G

DT_VALUES=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
SUBSTEPS_VALUES=(1 10 100)
SOLVER_TYPES=("explicit" "implicit")
MODEL_TYPES=("stable_neohookean" "linear")

source /home/trinityc/miniconda3/etc/profile.d/conda.sh
conda activate /home/trinityc/miniconda3/envs/genesis
cd /home/trinityc/testgenesis/src

for model in "${MODEL_TYPES[@]}"; do
    for dt in "${DT_VALUES[@]}"; do
        for substeps in "${SUBSTEPS_VALUES[@]}"; do
            for solver in "${SOLVER_TYPES[@]}"; do
                python fem_sweeper.py --dt ${dt} --substeps ${substeps} --model ${model} --solver ${solver}
            done
        done
    done
done