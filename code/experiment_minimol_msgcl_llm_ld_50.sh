#!/bin/bash

WORKDIR=".."
SCRIPT="$WORKDIR/code/minimol_triplet_runner.py"  
INPUT_CSV="data/ld50_zhu_150.csv"
OUTPUT_BASE="Outputs"

RUN_ID="ld50_llm_grid_sweep_llm_$(date +%Y%m%d_%H%M%S)"
TASK="ld50"  

cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"
TASK_DESC='Molecule lethally toxic to rats by oral exposure'
DATASET_DESC='Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. This measures rat lethal dose by oral exposure. Molecules that you will see have LD50'


python -u code/minimol_triplet_runner.py \
    --csv_path $INPUT_CSV \
    --output_dir $OUTPUT_BASE/$RUN_ID \
    --cache_file "/n/data1/hms/dbmi/farhat/anz226/Lyme_AZ/cache/smiles_fp_cache.pt" \
    --lrs 5e-4 \
    --epochs 25 \
    --hidden_sizes 1024 \
    --depths 3\
    --batch_size 128 \
    --margins "0.2" \
    --max_frags_per_mol 30\
    --replicates 20 \
    --combines "true" \
    --task "$TASK" \
    --bn_flags false\
    --triplet_weights 600\
    --repr_sizes 512\
    --is_regression\
    --min_score 3\
    --use_llm_plausibility\
    --max_llm_molecules 80\
    --task_description "$TASK_DESC" \
    --dataset_description "$DATASET_DESC" \
    --dropout 0.1\
    --get_two_sided\
    --max_neg_value 1.8\
    --task_description_negative "Non-Lethal Molecule With Low LD50"\

RUN_ID="ld50_llm_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE/$RUN_ID"


python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "" \
  --lrs 5e-4 \
  --epochs 25 \
  --hidden_sizes 512 \
  --depths 4\
  --batch_size 128 \
  --margins "0.2" \
  --max_frags_per_mol 30\
  --replicates 20 \
  --combines "false" \
  --task "$TASK" \
  --bn_flags false\
  --triplet_weights 600\
  --repr_sizes 512\
  --is_regression\
  --min_score 3\
  --dropout 0.1\

