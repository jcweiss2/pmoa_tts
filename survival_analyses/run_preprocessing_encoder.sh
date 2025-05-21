#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16g
#SBATCH -c1
#SBATCH --time=23:59:00
#SBATCH --output=sbatch_output/%A.out  # Store standard output in 'sbatch_output/' folder
#SBATCH --error=sbatch_error/%A.err    # Store errors in 'sbatch_error/' folder
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END  # Send email notifications
#SBATCH --mail-user=kumarsayantan94@gmail.com  # Replace with your actual email

module load python
# Load Conda properly
eval "$(conda shell.bash hook)"  # Ensures Conda is properly initialized
conda activate /data/kumars33/conda/envs/tts_forecasting

TIMES_OF_INTEREST=(0 24 168)

MODELS=(
  "google-bert/bert-base-uncased"
  "FacebookAI/roberta-base"
  "microsoft/deberta-v3-small"
  "answerdotai/ModernBERT-base"
  "answerdotai/ModernBERT-large"
)

DATASETS=("/data/CHARM-MIMIC/data/pmoa241217/Sayantan/")

TOTAL=$(( ${#DATASETS[@]} * ${#MODELS[@]} * ${#TIMES_OF_INTEREST[@]} ))
COUNT=1

for DATASET in "${DATASETS[@]}"; do 
  for MODEL in "${MODELS[@]}"; do
    for TIME in "${TIMES_OF_INTEREST[@]}"; do
      echo ""
      echo "[$COUNT/$TOTAL] Running preprocessing for:"
      echo "  → Dataset: $DATASET" 
      echo "  → Model:   $MODEL"
      echo "  → Time:    $TIME"
      echo "  → Date:    $(date '+%Y-%m-%d %H:%M:%S')"
      echo "-------------------------------------------"
      python preprocess_survival_data.py --data_folder "$DATASET" --model "$MODEL" --time_of_interest "$TIME"
      COUNT=$((COUNT + 1))
    done
  done
done

echo ""
echo "All preprocessing jobs completed!"


#python preprocess_survival_data.py --model google-bert/bert-base-uncased --time_of_interest 24