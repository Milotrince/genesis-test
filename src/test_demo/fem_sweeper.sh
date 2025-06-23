#!/bin/bash
#SBATCH --job-name=fem_sweeper
#SBATCH --output=logs/slurm_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=2G

# Define parameter combinations
DT_VALUES=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
SUBSTEPS_VALUES=(1 10 100)
SOLVER_TYPES=("explicit" "implicit")
MODEL_TYPES=("linear" "stable_neohookean")

# Function to submit job with retry
submit_with_retry() {
    local script_content="$1"
    local success=0
    while [ $success -eq 0 ]; do
        echo "$script_content" | sbatch
        if [ $? -eq 0 ]; then
            success=1
        else
            echo "sbatch failed, retrying in 5 seconds..."
            sleep 5
        fi
    done
}

# Submit jobs for all combinations
for model in "${MODEL_TYPES[@]}"; do
    for dt in "${DT_VALUES[@]}"; do
        for substeps in "${SUBSTEPS_VALUES[@]}"; do
            for solver in "${SOLVER_TYPES[@]}"; do
                script="#!/bin/bash
    #SBATCH --job-name=${model}_${solver}_${substeps}_${dt}
    #SBATCH --output=logs/${model}_${solver}_${substeps}_${dt}_%j.out
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --gres=gpu:1
    #SBATCH --time=1-00:00:00
    #SBATCH --mem=80G

    source /home/trinityc/miniconda3/etc/profile.d/conda.sh
    conda activate /home/trinityc/miniconda3/envs/genesis

    cd /home/trinityc/testgenesis/src
    python fem_sweeper.py \\
        --dt ${dt} \\
        --substeps ${substeps} \\
        --model ${model} \\
        --solver ${solver}"
                submit_with_retry "$script"
            done
        done
    done
done

# python fem_sweeper.py --dt 0.001 --substeps 10 --model stable_neohookean --solver explicit
