#!/bin/bash
# Author(s): Siqi Sun (siqi.sun@ed.ac.uk)
# How to use: 
# 1. cd the repo
# 2. $sbatch run_train_fe.sh config/fe/edi/fe_edi_r1_h384_lr5e-5.json /work/tc046/tc046/siqisun/exp/
#


# ====================
# Options for sbatch
# ====================

# The QoS specifies the limits to apply to your job.
# See https://cirrus.readthedocs.io/en/main/user-guide/batch.html#specifying-resources-in-job-scripts
#SBATCH --qos=XXXX

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:4

# GPU type
# any type is fine //SBATCH --constraint=a100_40

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=2-12:00:0

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=./slurm-llama-2_all_parallel-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=./slurm-llama-2_all_parallel-%A_%a.out

# RAM
#SBATCH --mem=200G

# Job array
#SBATCH --array=0-47

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# Load the required modules
#module load pytorch/1.12.1-gpu
module purge
module load baskerville
module load PyTorch
echo "modules are loaded"

. ../pytorch_env/bin/activate
echo "venv activated"

# echo "config_path: $1"
# echo "output_path: $2"

COMMAND="./run-llama-2_all_parallel.sh"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
