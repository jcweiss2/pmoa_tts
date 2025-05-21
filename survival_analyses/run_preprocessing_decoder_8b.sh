#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1,lscratch=128
#SBATCH --mem=16g
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

# TIME_OF_INTEREST=(0 24 168)

# MODELS=(
#   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#   "meta-llama/Llama-3.1-8B-Instruct"
#   "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#   "meta-llama/Llama-3.3-70B-Instruct"
#   )


python preprocess_survival_data_llm.py --model meta-llama/Llama-3.1-8B-Instruct --time_of_interest 24
python preprocess_survival_data_llm.py --model meta-llama/Llama-3.1-8B-Instruct --time_of_interest 0
python preprocess_survival_data_llm.py --model meta-llama/Llama-3.1-8B-Instruct --time_of_interest 168

python preprocess_survival_data_llm.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --time_of_interest 24
python preprocess_survival_data_llm.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --time_of_interest 0
python preprocess_survival_data_llm.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --time_of_interest 168






