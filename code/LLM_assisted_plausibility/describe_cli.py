import argparse
import json
import os
import sys

# Ensure this file can import sibling module when run directly
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(HERE, ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from LLM_assisted_plausibility.molecular_description_image import describe_molecules_batch  # type: ignore


def main():
    ap = argparse.ArgumentParser(description="Describe molecules via LLM (batch)")
    ap.add_argument("--smiles_json", type=str, required=True, help="Path to JSON file containing a list of SMILES strings")
    ap.add_argument("--output_json", type=str, required=True, help="Path to write JSON mapping {smiles: description}")
    ap.add_argument("--n_workers", type=int, default=20, help="Number of parallel workers")
    args = ap.parse_args()

    with open(args.smiles_json, "r") as f:
        smiles_list = json.load(f)

    if not isinstance(smiles_list, list):
        raise ValueError("smiles_json must contain a JSON list of strings")

    # Run description generation
    mapping = describe_molecules_batch(smiles_list=smiles_list, output_path=args.output_json, n_workers=int(args.n_workers))

    # Ensure output file exists and is complete
    try:
        with open(args.output_json, "w") as fout:
            json.dump(mapping, fout, indent=2)
    except Exception:
        # Fallback: if the function already wrote to output_json, leave it
        pass


if __name__ == "__main__":
    main()


