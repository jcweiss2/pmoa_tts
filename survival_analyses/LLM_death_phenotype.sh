#!/bin/bash
#SBATCH --array=0-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2,lscratch=128
#SBATCH --mem=8g
#SBATCH -c4
#SBATCH --time=23:59:00
#SBATCH --output=sbatch_output/%A.out  # Store standard output in 'sbatch_output/' folder
#SBATCH --error=sbatch_error/%A.err    # Store errors in 'sbatch_error/' folder
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END  # Send email notifications
#SBATCH --mail-user=kumarsayantan94@gmail.com  # Replace with your actual email

module load python
# Load Conda properly
eval "$(conda shell.bash hook)"  # Ensures Conda is properly initialized
conda activate /data/kumars33/conda/envs/tts_forecasting

python /data/CHARM-MIMIC/data/pmoa241217/Sayantan/LLM_death_phenotype_25k.py

#sbatch --array=0-5 --partition=gpu --gres=gpu:a100:2,lscratch:128 --mem=8g -c4 --time 23:59:00 --wrap=“source myconda; conda activate for_llm; 
#python /data/weissjc/workspace/stts/scripts/tts_forecast/apply_llm_to_phenotype_death_t2s2.py”