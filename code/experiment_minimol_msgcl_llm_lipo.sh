#!/bin/bash

WORKDIR=".."
SCRIPT="$WORKDIR/code/minimol_triplet_runner.py"  
INPUT_CSV="data/lipophilicity_150_train.csv"
OUTPUT_BASE="Outputs"

RUN_ID="experiment_lipophilicity_two_sided$(date +%Y%m%d_%H%M%S)"
TASK="lipophilicity"  


cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"
TASK_DESC='Lipophilic Molecule'
DATASET_DESC='This dataset, curated from ChEMBL database, provides experimental results of octanol/water distribution coefficient (log D at pH 7.4) of 4200 compoundss'


python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "" \
  --lrs 3e-4 \
  --epochs 100 \
  --hidden_sizes 128 \
  --depths 3\
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
  --use_llm_plausibility\
  --max_llm_molecules 60\
  --task_description "$TASK_DESC" \
  --dataset_description "$DATASET_DESC" \
  --dropout 0.1\
  --get_two_sided\
  --max_neg_value 0\
  --task_description_negative "Hydrophilic Molecule"\

RUN_ID="experiment_lipophilicity_baseline_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_BASE/$RUN_ID"


  python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "/n/data1/hms/dbmi/farhat/anz226/Lyme_AZ/cache/smiles_fp_cache.pt" \
  --lrs 5e-4 \
  --epochs 200 \
  --hidden_sizes 512 \
  --depths 4\
  --batch_size 128 \
  --margins "0.2" \
  --max_frags_per_mol 30\
  --replicates 20 \
  --combines "true" \
  --task "$TASK" \
  --bn_flags true\
  --triplet_weights 600\
  --repr_sizes 512\
  --is_regression\
  --min_score 3\
  --dropout 0.1\

