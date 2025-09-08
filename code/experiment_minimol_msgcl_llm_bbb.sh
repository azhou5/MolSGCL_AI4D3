
WORKDIR=".."
SCRIPT="$WORKDIR/code/minimol_triplet_runner.py"  
INPUT_CSV="data/bbb_martins_150.csv"
OUTPUT_BASE="Outputs"

RUN_ID="experiment_bbb_llm_one_sided_$(date +%Y%m%d_%H%M%S)"
TASK="bbb"  # choices: lipophilicity | ames


cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"
TASK_DESC='Blood Brain Barrier Permeator'
DATASET_DESC='The BBB, or blood–brain barrier, is the specialized physiological barrier that separates circulating blood from the brain’s extracellular fluid. The positive label in this dataset are molecules that can cross the BBB, while the negative label are molecules that cannot cross the blood brain barrier.'

python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "" \
  --lrs 3e-4 \
  --epochs 100 \
  --hidden_sizes 1024 \
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
  --use_llm_plausibility\
  --max_llm_molecules 80\
  --task_description "$TASK_DESC" \
  --dataset_description "$DATASET_DESC" \
  --dropout 0.1\
  --get_two_sided\
  --task_description_negative "Blood Brain Barrier Non-Permeator (Does not Pass)"\


RUN_ID="bbb_llm_baseline_$(date +%Y%m%d_%H%M%S)"
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
  --bn_flags true\
  --triplet_weights 600\
  --repr_sizes 512\
  --dropout 0.1\

 # --min_score 3\


 # --get_two_sided\
  #--max_neg_value -1\
