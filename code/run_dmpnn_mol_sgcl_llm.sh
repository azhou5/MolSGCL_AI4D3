
WORKDIR=".."
SCRIPT="$WORKDIR/code/dmpnn_mol_sgcl.py"
OUTPUT_BASE="Outputs/"
INPUT_CSV="data/lipophilicity_150_train.csv"

# Metadata
RUN_ID="chemprop_experiment_lipophilicity_sgcl_dmpnn_llm_1e_3_300epoch$(date +%Y%m%d_%H%M%S)"

cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"

python -u "code/dmpnn_mol_sgcl.py" \
  --run_id "$RUN_ID" \
  --input_csv_path "$INPUT_CSV" \
  --output_dir "$OUTPUT_BASE"\
  --is_regression \
  --task "lipophilicity"\
  --max_molecules_for_plausibility 100 \
  --total_epochs "300" \
  --reinterpretations "5" \
  --learning_rates "1e-3" \
  --triplet_weights "40" \
  --margins "0.25"\
  --n_replicates "5"\
  --use_llm_plausibility \
  --min_value_regression 3\
  --task_description "Lipophilic Molecule" \
  --dataset_description "Lipophilicity is a measure of how easily a chemical dissolves in a non-polar solvent." \


