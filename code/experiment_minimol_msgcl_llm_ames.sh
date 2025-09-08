#!/bin/bash
WORKDIR=".."
SCRIPT="$WORKDIR/code/minimol_triplet_runner.py"  
INPUT_CSV="data/ames_150.csv"
OUTPUT_BASE="Outputs"

RUN_ID="experiment_ames_llm_400_one_sided$(date +%Y%m%d_%H%M%S)"
TASK="ames"  # choices: lipophilicity | ames


cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"


python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "" \
  --lrs 5e-4 \
  --epochs 25 \
  --hidden_sizes 1024 \
  --depths 4\
  --batch_size 128 \
  --margins "0.2" \
  --max_frags_per_mol 30\
  --replicates 20 \
  --combines "true" \
  --task "$TASK" \
  --bn_flags false\
  --triplet_weights 400\
  --repr_sizes 512\
  --use_llm_plausibility\
  --max_llm_molecules 80\
  --task_description "Mutagen (By AMES Assay)" \
  --dataset_description "The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations. The dataset is aggregated from four papers. A positive test indicates that the chemical is mutagenic either by itself or when activated with S9 rat liver fraction. The Ames test uses several strains of the bacterium Salmonella typhimurium that carry mutations in genes involved in histidine synthesis." \
  --dropout 0.1\


RUN_ID="experiment_ames_llm_baseline$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE/$RUN_ID"


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
  --dropout 0.1\

