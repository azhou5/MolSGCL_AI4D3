

WORKDIR=".."
SCRIPT="$WORKDIR/code/minimol_triplet_runner.py"  
INPUT_CSV="data/freesolv_150_train_val_from_test_minTestSim.csv"
OUTPUT_BASE="Outputs"

RUN_ID="experiment_freesolv_llm_$(date +%Y%m%d_%H%M%S)"
TASK="freesolv"  


cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"
TASK_DESC='Insoluable Molecule'
DATASET_DESC="The Free Solvation Database, FreeSolv(SAMPL), provides experimental hydration free energy of small molecules in water. Energies cover a range of approximately 29 kcal/mol, from 3.43 kcal/mol … to −25.47 kcal/mol …"


python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "" \
  --lrs 5e-4 \
  --epochs 200 \
  --hidden_sizes 128 \
  --depths 3\
  --batch_size 128 \
  --margins "0.2" \
  --max_frags_per_mol 30\
  --replicates 20 \
  --combines "true" \
  --task "$TASK" \
  --is_regression\
  --min_score -3\
  --bn_flags true\
  --triplet_weights 600\
  --repr_sizes 512\
  --use_llm_plausibility\
  --max_llm_molecules 80\
  --task_description "$TASK_DESC" \
  --dataset_description "$DATASET_DESC" \
  --dropout 0.1\
  --max_neg_value -7\
  --get_two_sided\
  --task_description_negative "Soluble Molecule"\

