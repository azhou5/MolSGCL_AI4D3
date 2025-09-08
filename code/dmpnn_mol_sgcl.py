import argparse
import os
import random
import re
import copy
import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))        
PARENT = os.path.abspath(os.path.join(BASE, ".."))       
sys.path.insert(0, PARENT)

from chemprop_custom.data import MoleculeDatapoint, MoleculeDataset

import numpy as np
import pandas as pd
import torch
import concurrent.futures
import csv
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity

# Local project path (ensure absolute imports work when run as a script)


from chemprop_custom.data import MoleculeDatapoint, MoleculeDataset
from chemprop_custom.data.dataloader import build_dataloader
from chemprop_custom.featurizers import (
    SimpleMoleculeMolGraphFeaturizer,
    MultiHotAtomFeaturizer,
)
from chemprop_custom import featurizers
from chemprop_custom.models.model import MPNN, train_triplet_encoder_bce

from get_rationale_for_plausibility import run_MCTS
from get_rationale_regression import run_MCTS as run_MCTS_regression

from plausibility_utils import (
    classify_logp_rationales,
    classify_ames_rationales)

from model_utils import get_pred_chemprop, run_chemprop_training

# LLM-assisted plausibility utilities
from LLM_assisted_plausibility.get_plausibility import get_plausibility
from LLM_assisted_plausibility.molecular_description_image import describe_molecules_batch


# ----------------------------
# Utility metrics
# ----------------------------

def calculate_auroc_from_preds(predictions, targets):
    try:
        predictions = np.array(predictions).reshape(-1)
        targets = np.array(targets).reshape(-1)
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(targets, predictions))
    except Exception:
        return 0.0


def calculate_rmse_from_preds(predictions, targets):
    try:
        predictions = np.array(predictions).reshape(-1)
        targets = np.array(targets).reshape(-1)
        from sklearn.metrics import mean_squared_error

        return float(np.sqrt(mean_squared_error(targets, predictions)))
    except Exception:
        return float('inf')



def is_valid_molecule(smiles):
    if smiles is None or not isinstance(smiles, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False



def get_cleaned_fragments(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    fragments = set()
    try:
        fragments.update(BRICS.BRICSDecompose(mol))
    except Exception:
        pass

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
        if scaffold_smiles:
            fragments.add(scaffold_smiles)
    except Exception:
        pass

    valid_fragments = []
    for frag_smiles in fragments:
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol is not None and 5 <= frag_mol.GetNumAtoms() <= 12:
            valid_fragments.append(frag_smiles)

    cleaned_fragments = []
    for fragment in valid_fragments:
        cleaned_fragment = re.sub(r'\[\d+\*\]', '', fragment).replace('()', '')
        cleaned_fragments.append(cleaned_fragment)

    return cleaned_fragments


# ----------------------------
# Datapoint preparation
# ----------------------------

def create_datapoint_with_rationales(smiles: str, rationale_0: str, rationale_1: str, target: float) -> MoleculeDatapoint:
    featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    mol = Chem.MolFromSmiles(smiles)
    mol_graph = featurizer(mol)
    rat_0_mol = Chem.MolFromSmiles(rationale_0)
    rat_1_mol = Chem.MolFromSmiles(rationale_1)
    rat_0_graph = featurizer(rat_0_mol)
    rat_1_graph = featurizer(rat_1_mol)

    return MoleculeDatapoint(
        mol=mol,
        V_d=None,
        y=np.array([float(target)]),
        weight=1.0,
        lt_mask=None,
        gt_mask=None,
        mg=mol_graph,
        rationale_mg=rat_0_graph,
        neg_rationale_mg=rat_1_graph,
    )


def prepare_datapoints_without_rationales(dataset_df, target_column_name):
    datapoints = []
    featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    for _, row in dataset_df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue
        mol_graph = featurizer(mol)
        dp = MoleculeDatapoint(
            mol=mol,
            V_d=None,
            y=np.array([float(row[target_column_name])]),
            weight=1.0,
            lt_mask=None,
            gt_mask=None,
            mg=mol_graph,
            rationale_mg=None,
            neg_rationale_mg=None,
        )
        datapoints.append(dp)
    return datapoints


def get_rationales_for_smiles(data, smiles):
    entry = data.get(smiles, None)
    if entry is None:
        return "None", "None", "not_found"

    rationales = entry['Rationales']
    plausibility = np.array(entry['plausibility'])

    if 1 in plausibility and -1 in plausibility:
        plausible_idx = np.where(plausibility == 1)[0][0]
        implausible_idx = np.where(plausibility == -1)[0][0]
        return rationales[plausible_idx], rationales[implausible_idx], "standard_pair"

    if np.sum(plausibility == -1) == 1 and np.all((plausibility == 0) | (plausibility == -1)):
        implausible_idx = np.where(plausibility == -1)[0][0]
        positive_candidate_indices = np.where(plausibility == 0)[0]
        if len(positive_candidate_indices) > 0:
            plausible_idx = random.choice(positive_candidate_indices)
            return rationales[plausible_idx], rationales[implausible_idx], "random_positive"
        return "None", "None", "skipped_no_positive_candidate"

    if np.all(plausibility == 0):
        return "None", "None", "all_zeros"

    return "None", "None", "unhandled_case"


def prepare_rationale_data(rationales_df, smiles_to_rationale_plausibility, full_dataset_df, target_column_name):
    datapoints = []
    smiles_with_plausibility = set(smiles_to_rationale_plausibility.keys())
    stats = {
        "success_standard_pair": 0,
        "success_random_positive": 0,
        "skipped_no_rationales": 0,
        "skipped_all_zeros": 0,
        "skipped_unhandled": 0,
    }

    rationales_df = rationales_df.drop_duplicates(subset=['SMILES'])
    for _, row in rationales_df.iterrows():
        s = row['SMILES']
        if s not in smiles_with_plausibility:
            continue

        plausible_rationale, implausible_rationale, status = get_rationales_for_smiles(
            smiles_to_rationale_plausibility, s
        )

        if status == "all_zeros":
            stats["skipped_all_zeros"] += 1
            continue

        if status in ["not_found", "skipped_no_positive_candidate", "unhandled_case"] or plausible_rationale == "None" or implausible_rationale == "None":
            if status == "unhandled_case":
                stats["skipped_unhandled"] += 1
            stats["skipped_no_rationales"] += 1
            continue

        if status == "standard_pair":
            stats["success_standard_pair"] += 1
        elif status == "random_positive":
            stats["success_random_positive"] += 1

        dp = create_datapoint_with_rationales(
            smiles=s,
            rationale_0=plausible_rationale,
            rationale_1=implausible_rationale,
            target=float(row[target_column_name]),
        )
        datapoints.append(dp)

    return datapoints, stats


# ----------------------------
# Rationale generation + plausibility
# ----------------------------

def compute_rationales_and_plausibility(model, train_df, property_name, output_dir, cycle_num, combo_idx, is_regression=False, min_value_regression=0.5, max_molecules_for_plausibility=50, use_llm_plausibility=False, task_description='', dataset_description=''):
    if is_regression:
        top_df = train_df[train_df['Y'] > min_value_regression].copy()
        if len(top_df) == 0:
            return {}, pd.DataFrame()
        num_to_select = min(len(top_df), max_molecules_for_plausibility)
        selected_df = top_df.sample(n=num_to_select).reset_index(drop=True)
        selected_smiles = selected_df['SMILES'].tolist()

        rationales_df = run_MCTS_regression(
            selected_smiles,
            model=model,
            disable_multiprocessing=True,
            property_name=property_name,
            output_dir=output_dir,
            prop_delta_threshold=-10,
        )
    else:
        top_df = train_df[train_df['Y'] == 1].copy()
        if len(top_df) == 0:
            return {}, pd.DataFrame()
        num_to_select = min(len(top_df), max_molecules_for_plausibility)
        selected_df = top_df.sample(n=num_to_select).reset_index(drop=True)
        selected_smiles = selected_df['SMILES'].tolist()

        rationales_df = run_MCTS(
            selected_smiles,
            model=model,
            disable_multiprocessing=True,
            property_name=property_name,
            output_dir=output_dir,
        )

    if rationales_df.empty:
        return {}, pd.DataFrame()

    smiles_to_rationales = {}
    rationales_df.rename(columns={'smiles': 'SMILES'}, inplace=True)
    for _, row in rationales_df.iterrows():
        smiles = row['SMILES']
        rationale_cols = [c for c in row.index if isinstance(c, str) and c.startswith('rationale_')]
        column_rationales = [str(row[c]) for c in rationale_cols if pd.notna(row[c])]
        cleaned_column_rationales = [re.sub(r"\[\d+\*\]", "", r).replace("()", "") for r in column_rationales]

        extra_frags = get_cleaned_fragments(smiles)
        fpgen = AllChem.GetMorganGenerator(radius=3)
        existing_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(c)) for c in cleaned_column_rationales if Chem.MolFromSmiles(c)]

        filtered_frags = []
        for frag in extra_frags:
            m_frag = Chem.MolFromSmiles(frag)
            if not m_frag:
                continue
            fp_frag = fpgen.GetFingerprint(m_frag)
            if not any(TanimotoSimilarity(fp_frag, e) >= 0.3 for e in existing_fps if fp_frag and e):
                existing_fps.append(fp_frag)
                filtered_frags.append(frag)

        combined_candidates = cleaned_column_rationales + filtered_frags
        seen = set()
        valid_candidates = []
        for cand in combined_candidates:
            if not is_valid_molecule(cand):
                continue
            if cand in seen:
                continue
            seen.add(cand)
            valid_candidates.append(cand)

        if len(valid_candidates) < 3:
            continue

        try:
            preds = get_pred_chemprop(model, valid_candidates)
        except Exception:
            continue

        pairs = [(valid_candidates[i], preds[i]) for i in range(min(len(valid_candidates), len(preds)))]
        pairs.sort(key=lambda x: float(x[1]), reverse=True)
        if len(pairs) > 4:
            top_two = [pairs[0][0], pairs[1][0]]
            remaining = [s for s, _ in pairs[2:]]
            sampled = random.sample(remaining, 2) if len(remaining) >= 2 else remaining
            selected_rationales = top_two + sampled
        else:
            selected_rationales = [s for s, _ in pairs[:4]]

        smiles_to_rationales[str(smiles)] = selected_rationales

    data_tuples = [(k, s, train_df[train_df['SMILES'] == k]['Y'].iloc[0]) for k, v in smiles_to_rationales.items() for s in v[:6]]
    rationales_df_processed = pd.DataFrame(data_tuples, columns=['SMILES', 'Rationales', 'Y']).drop_duplicates()
    rationales_df_processed['is_valid'] = rationales_df_processed['Rationales'].apply(is_valid_molecule)
    rationales_df_processed = rationales_df_processed[rationales_df_processed['is_valid']].drop(columns='is_valid', errors='ignore')
    smiles_counts = rationales_df_processed['SMILES'].value_counts()
    valid_smiles = smiles_counts[smiles_counts >= 3].index
    rationales_df_processed = rationales_df_processed[rationales_df_processed['SMILES'].isin(valid_smiles)]

    smiles_to_rationales_container = {}
    for smiles_val in valid_smiles:
        rationales_list = rationales_df_processed[rationales_df_processed['SMILES'] == smiles_val]['Rationales'].tolist()
        smiles_to_rationales_container[smiles_val] = {'Rationales': rationales_list}

    # Assign plausibility scores
    if use_llm_plausibility:
        # Use LLM to compute plausibility. Describe molecules in batches per parent SMILES.
        tasks = []
        for k, v in smiles_to_rationales_container.items():
            all_smiles = v['Rationales'] + [k]
            try:
                descriptions = describe_molecules_batch(all_smiles, n_workers=20)
            except Exception:
                descriptions = {s: '' for s in all_smiles}
            v_descriptions = [descriptions.get(r, '') for r in v['Rationales']]
            original_desc = descriptions.get(k, '')
            tasks.append((k, v_descriptions, v['Rationales'], original_desc))

        # Parallelize plausibility calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_map = {
                executor.submit(
                    get_plausibility,
                    description_list=desc_list,
                    smiles_list=rat_list,
                    original_smiles=mol_smiles,
                    original_molecule_description=orig_desc,
                    task_description=task_description,
                    dataset_description=dataset_description,
                ): mol_smiles
                for (mol_smiles, desc_list, rat_list, orig_desc) in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                key = future_map[future]
                try:
                    result = future.result()
                except Exception:
                    # Fallback to zeros if LLM call fails
                    num_rationales = len(smiles_to_rationales_container[key]['Rationales'])
                    result = [0] * num_rationales
                smiles_to_rationales_container[key]['plausibility'] = result
    else:
        for k, v in smiles_to_rationales_container.items():
            plausibility_array = plausibility_fn(smiles_list=v['Rationales'])
            smiles_to_rationales_container[k]['plausibility'] = plausibility_array

    return smiles_to_rationales_container, rationales_df_processed


def compute_negative_control_rationales(model, train_df, property_name, output_dir, cycle_num, combo_idx, is_regression=False, min_value_regression=0.5, max_molecules_for_plausibility=50):
    # This mirrors compute_rationales_and_plausibility but assigns random plausibility labels.
    if is_regression:
        top_df = train_df[train_df['Y'] > min_value_regression].copy()
        if len(top_df) == 0:
            return {}, pd.DataFrame()
        num_to_select = min(len(top_df), max_molecules_for_plausibility)
        selected_df = top_df.sample(n=num_to_select).reset_index(drop=True)
        selected_smiles = selected_df['SMILES'].tolist()

        rationales_df = run_MCTS_regression(
            selected_smiles,
            model=model,
            disable_multiprocessing=True,
            property_name=property_name,
            output_dir=output_dir,
            prop_delta_threshold=-10,
        )
    else:
        top_df = train_df[train_df['Y'] == 1].copy()
        if len(top_df) == 0:
            return {}, pd.DataFrame()
        num_to_select = min(len(top_df), max_molecules_for_plausibility)
        selected_df = top_df.sample(n=num_to_select).reset_index(drop=True)
        selected_smiles = selected_df['SMILES'].tolist()

        rationales_df = run_MCTS(
            selected_smiles,
            model=model,
            disable_multiprocessing=True,
            property_name=property_name,
            output_dir=output_dir,
        )

    if rationales_df.empty:
        return {}, pd.DataFrame()

    smiles_to_rationales = {}
    rationales_df.rename(columns={'smiles': 'SMILES'}, inplace=True)
    for _, row in rationales_df.iterrows():
        smiles = row['SMILES']
        rationale_cols = [c for c in row.index if isinstance(c, str) and c.startswith('rationale_')]
        column_rationales = [str(row[c]) for c in rationale_cols if pd.notna(row[c])]
        cleaned_column_rationales = [re.sub(r"\[\d+\*\]", "", r).replace("()", "") for r in column_rationales]

        extra_frags = get_cleaned_fragments(smiles)
        fpgen = AllChem.GetMorganGenerator(radius=3)
        existing_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(c)) for c in cleaned_column_rationales if Chem.MolFromSmiles(c)]

        filtered_frags = []
        for frag in extra_frags:
            m_frag = Chem.MolFromSmiles(frag)
            if not m_frag:
                continue
            fp_frag = fpgen.GetFingerprint(m_frag)
            if not any(TanimotoSimilarity(fp_frag, e) >= 0.3 for e in existing_fps if fp_frag and e):
                existing_fps.append(fp_frag)
                filtered_frags.append(frag)

        combined_candidates = cleaned_column_rationales + filtered_frags
        seen = set()
        valid_candidates = []
        for cand in combined_candidates:
            if not is_valid_molecule(cand):
                continue
            if cand in seen:
                continue
            seen.add(cand)
            valid_candidates.append(cand)

        if len(valid_candidates) < 3:
            continue

        try:
            preds = get_pred_chemprop(model, valid_candidates)
        except Exception:
            continue

        pairs = [(valid_candidates[i], preds[i]) for i in range(min(len(valid_candidates), len(preds)))]
        pairs.sort(key=lambda x: float(x[1]), reverse=True)
        if len(pairs) > 4:
            top_two = [pairs[0][0], pairs[1][0]]
            remaining = [s for s, _ in pairs[2:]]
            sampled = random.sample(remaining, 2) if len(remaining) >= 2 else remaining
            selected_rationales = top_two + sampled
        else:
            selected_rationales = [s for s, _ in pairs[:4]]

        smiles_to_rationales[str(smiles)] = selected_rationales

    data_tuples = [(k, s, train_df[train_df['SMILES'] == k]['Y'].iloc[0]) for k, v in smiles_to_rationales.items() for s in v[:6]]
    rationales_df_processed = pd.DataFrame(data_tuples, columns=['SMILES', 'Rationales', 'Y']).drop_duplicates()
    rationales_df_processed['is_valid'] = rationales_df_processed['Rationales'].apply(is_valid_molecule)
    rationales_df_processed = rationales_df_processed[rationales_df_processed['is_valid']].drop(columns='is_valid', errors='ignore')
    smiles_counts = rationales_df_processed['SMILES'].value_counts()
    valid_smiles = smiles_counts[smiles_counts >= 3].index
    rationales_df_processed = rationales_df_processed[rationales_df_processed['SMILES'].isin(valid_smiles)]

    smiles_to_rationales_container = {}
    for smiles_val in valid_smiles:
        rationales_list = rationales_df_processed[rationales_df_processed['SMILES'] == smiles_val]['Rationales'].tolist()
        smiles_to_rationales_container[smiles_val] = {'Rationales': rationales_list}

    # Assign random plausibility as negative control
    for k, v in smiles_to_rationales_container.items():
        num_rationales = len(v['Rationales'])
        plausibility_array = [0] * num_rationales
        if num_rationales >= 2:
            indices = random.sample(range(num_rationales), 2)
            plausibility_array[indices[0]] = 1
            plausibility_array[indices[1]] = -1
        elif num_rationales == 1:
            plausibility_array[0] = random.choice([1, -1])
        smiles_to_rationales_container[k]['plausibility'] = plausibility_array

    return smiles_to_rationales_container, rationales_df_processed


# ----------------------------
# CSV live writing helpers
# ----------------------------

def append_row_to_csv(file_path, row_dict):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    # Maintain a consistent column order based on current row keys
    fieldnames = list(row_dict.keys())
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def append_rows_to_csv(file_path, rows, fieldnames=None):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if rows is None or len(rows) == 0:
        return
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ----------------------------
# Training loop
# ----------------------------

def run_multi_cycle_training(train_df, val_df, test_df, n_cycles, triplet_epochs, triplet_lr, triplet_weight, margin, property_name, output_dir, combo_idx, is_regression=False, min_value_regression=0.5, max_molecules_for_plausibility=50, use_llm_plausibility=False, task_description='', dataset_description='', live_results_path=None, replicate_idx: int = 1, plausibility_log_path: str = None):
    # Minimal initial supervised warmup
    warmup = run_chemprop_training(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        smiles_column="SMILES",
        target_columns=["Y"],
        max_epochs=2,
        init_lr=1e-8,
        max_lr=1e-8,
        final_lr=1e-8,
        accelerator="cpu",
        devices=1,
        predictor="regression" if is_regression else "classification",
    )
    current_model = warmup['model']
    current_no_triplet_model = copy.deepcopy(current_model)
    current_negative_control_model = copy.deepcopy(current_model)

    all_cycle_results = []
    for cycle in range(n_cycles):
        smiles_to_plausibility, rationales_df = compute_rationales_and_plausibility(
            model=current_model,
            train_df=train_df,
            property_name=property_name,
            output_dir=output_dir,
            cycle_num=cycle + 1,
            combo_idx=combo_idx,
            is_regression=is_regression,
            min_value_regression=min_value_regression,
            max_molecules_for_plausibility=max_molecules_for_plausibility,
            use_llm_plausibility=use_llm_plausibility,
            task_description=task_description,
            dataset_description=dataset_description,
        )
        smiles_to_plausibility_nc, rationales_df_nc = compute_negative_control_rationales(
            model=current_negative_control_model,
            train_df=train_df,
            property_name=property_name,
            output_dir=output_dir,
            cycle_num=cycle + 1,
            combo_idx=combo_idx,
            is_regression=is_regression,
            min_value_regression=min_value_regression,
            max_molecules_for_plausibility=max_molecules_for_plausibility,
        )

        # Log plausibility ratings if requested
        if plausibility_log_path is not None:
            try:
                rows_to_log = []
                for parent_smiles, payload in smiles_to_plausibility.items():
                    rats = payload.get('Rationales', [])
                    plaus = payload.get('plausibility', [])
                    for i, r in enumerate(rats):
                        rows_to_log.append({
                            'replicate': replicate_idx,
                            'combo_idx': combo_idx,
                            'cycle': cycle + 1,
                            'branch': 'sgcl',
                            'parent_smiles': parent_smiles,
                            'rationale_smiles': r,
                            'plausibility': plaus[i] if i < len(plaus) else None,
                        })
                for parent_smiles, payload in smiles_to_plausibility_nc.items():
                    rats = payload.get('Rationales', [])
                    plaus = payload.get('plausibility', [])
                    for i, r in enumerate(rats):
                        rows_to_log.append({
                            'replicate': replicate_idx,
                            'combo_idx': combo_idx,
                            'cycle': cycle + 1,
                            'branch': 'neg_control',
                            'parent_smiles': parent_smiles,
                            'rationale_smiles': r,
                            'plausibility': plaus[i] if i < len(plaus) else None,
                        })
                append_rows_to_csv(plausibility_log_path, rows_to_log)
            except Exception:
                pass

        train_datapoints_with_rationales, _ = ([], {})
        if smiles_to_plausibility:
            train_datapoints_with_rationales, _ = prepare_rationale_data(rationales_df, smiles_to_plausibility, train_df, "Y")
        molecules_with_rationales = {Chem.MolToSmiles(dp.mol) for dp in train_datapoints_with_rationales}
        train_molecules_without_rationales = train_df[~train_df['SMILES'].isin(molecules_with_rationales)]
        train_datapoints_without_rationales = prepare_datapoints_without_rationales(train_molecules_without_rationales, "Y")

        # Negative control datasets
        train_datapoints_with_rationales_nc, _ = ([], {})
        if smiles_to_plausibility_nc:
            train_datapoints_with_rationales_nc, _ = prepare_rationale_data(rationales_df_nc, smiles_to_plausibility_nc, train_df, "Y")
        molecules_with_rationales_nc = {Chem.MolToSmiles(dp.mol) for dp in train_datapoints_with_rationales_nc}
        train_molecules_without_rationales_nc = train_df[~train_df['SMILES'].isin(molecules_with_rationales_nc)]
        train_datapoints_without_rationales_nc = prepare_datapoints_without_rationales(train_molecules_without_rationales_nc, "Y")

        featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())
        train_dataset = MoleculeDataset(train_datapoints_with_rationales + train_datapoints_without_rationales, featurizer)
        val_dataset = MoleculeDataset(prepare_datapoints_without_rationales(val_df, "Y"), featurizer)
        class_balance = False if is_regression else True
        train_loader = build_dataloader(train_dataset, shuffle=True, class_balance=class_balance, rationale_balance=False, batch_size=128)
        val_loader = build_dataloader(val_dataset, shuffle=False, class_balance=class_balance, rationale_balance=False, batch_size=128)

        trained_model = train_triplet_encoder_bce(
            current_model,
            train_loader,
            val_loader,
            max_epochs=triplet_epochs,
            margin=margin,
            triplet_weight=triplet_weight,
            init_lr=triplet_lr,
            max_lr=triplet_lr,
            final_lr=triplet_lr,
            is_regression=is_regression,
        )
        # Baseline: no-triplet
        base_train_dps = prepare_datapoints_without_rationales(train_df, "Y")
        base_val_dps = prepare_datapoints_without_rationales(val_df, "Y")
        base_featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())
        train_dataset_no_triplet = MoleculeDataset(base_train_dps, base_featurizer)
        val_dataset_no_triplet = MoleculeDataset(base_val_dps, base_featurizer)
        class_balance_no_triplet = False if is_regression else True
        train_loader_no_triplet = build_dataloader(train_dataset_no_triplet, shuffle=True, class_balance=class_balance_no_triplet, rationale_balance=False, batch_size=128)
        val_loader_no_triplet = build_dataloader(val_dataset_no_triplet, shuffle=False, class_balance=False, rationale_balance=False, batch_size=128)
        no_triplet_model = train_triplet_encoder_bce(
            current_no_triplet_model,
            train_loader_no_triplet,
            val_loader_no_triplet,
            max_epochs=triplet_epochs,
            margin=margin,
            triplet_weight=0.0,
            init_lr=triplet_lr,
            max_lr=triplet_lr,
            final_lr=triplet_lr,
            is_regression=is_regression,
        )

        # Negative control branch: random plausibility
        train_dataset_nc = MoleculeDataset(train_datapoints_with_rationales_nc + train_datapoints_without_rationales_nc, featurizer)
        val_dataset_nc = MoleculeDataset(prepare_datapoints_without_rationales(val_df, "Y"), featurizer)
        train_loader_nc = build_dataloader(train_dataset_nc, shuffle=True, class_balance=class_balance, rationale_balance=False, batch_size=128)
        val_loader_nc = build_dataloader(val_dataset_nc, shuffle=False, class_balance=class_balance, rationale_balance=False, batch_size=128)
        neg_control_model = train_triplet_encoder_bce(
            current_negative_control_model,
            train_loader_nc,
            val_loader_nc,
            max_epochs=triplet_epochs,
            margin=margin,
            triplet_weight=triplet_weight,
            init_lr=triplet_lr,
            max_lr=triplet_lr,
            final_lr=triplet_lr,
            is_regression=is_regression,
        )

        # Evaluate
        current_model = trained_model
        current_no_triplet_model = no_triplet_model
        current_negative_control_model = neg_control_model

        test_preds = get_pred_chemprop(trained_model, test_df['SMILES'].tolist())
        metric_test = calculate_rmse_from_preds(test_preds, test_df['Y'].tolist()) if is_regression else calculate_auroc_from_preds(test_preds, test_df['Y'].tolist())
        val_preds = get_pred_chemprop(trained_model, val_df['SMILES'].tolist())
        metric_val = calculate_rmse_from_preds(val_preds, val_df['Y'].tolist()) if is_regression else calculate_auroc_from_preds(val_preds, val_df['Y'].tolist())

        test_preds_no_triplet = get_pred_chemprop(no_triplet_model, test_df['SMILES'].tolist())
        no_triplet_metric_test = calculate_rmse_from_preds(test_preds_no_triplet, test_df['Y'].tolist()) if is_regression else calculate_auroc_from_preds(test_preds_no_triplet, test_df['Y'].tolist())
        val_preds_no_triplet = get_pred_chemprop(no_triplet_model, val_df['SMILES'].tolist())
        no_triplet_metric_val = calculate_rmse_from_preds(val_preds_no_triplet, val_df['Y'].tolist()) if is_regression else calculate_auroc_from_preds(val_preds_no_triplet, val_df['Y'].tolist())

        test_preds_nc = get_pred_chemprop(neg_control_model, test_df['SMILES'].tolist())
        neg_control_metric_test = calculate_rmse_from_preds(test_preds_nc, test_df['Y'].tolist()) if is_regression else calculate_auroc_from_preds(test_preds_nc, test_df['Y'].tolist())
        val_preds_nc = get_pred_chemprop(neg_control_model, val_df['SMILES'].tolist())
        neg_control_metric_val = calculate_rmse_from_preds(val_preds_nc, val_df['Y'].tolist()) if is_regression else calculate_auroc_from_preds(val_preds_nc, val_df['Y'].tolist())


        cycle_result = {
            'combo_idx': combo_idx,
            'cycle': cycle + 1,
            'n_cycles': n_cycles,
            'rmse_test' if is_regression else 'auroc_test': metric_test,
            'rmse_val' if is_regression else 'auroc_val': metric_val,
            'num_rationales': len(train_datapoints_with_rationales),
            'num_molecules_with_rationales': len(smiles_to_plausibility),
            'training_status': 'success',
            'hyperparams': {
                'triplet_epochs': triplet_epochs,
                'triplet_lr': triplet_lr,
                'triplet_weight': triplet_weight,
                'margin': margin,
            },
            'replicate': replicate_idx,
        }
        # Augment with baseline and negative-control metrics
        cycle_result.update({
            f'no_triplet_{"rmse" if is_regression else "auroc"}_test': no_triplet_metric_test,
            f'no_triplet_{"rmse" if is_regression else "auroc"}_val': no_triplet_metric_val,
            f'neg_control_{"rmse" if is_regression else "auroc"}_test': neg_control_metric_test,
            f'neg_control_{"rmse" if is_regression else "auroc"}_val': neg_control_metric_val,
            'neg_control_num_rationales': len(train_datapoints_with_rationales_nc),
            'neg_control_num_molecules_with_rationales': len(smiles_to_plausibility_nc),
        })
        all_cycle_results.append(cycle_result)
        # Live-append this cycle's results if requested
        if live_results_path is not None:
            try:
                append_row_to_csv(live_results_path, cycle_result)
            except Exception:
                pass

    return all_cycle_results


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SGCL D-MPNN experiment with rationale-guided training.")
    parser.add_argument("--run_id", type=str, default="mol_sgcl_default", help="Unique identifier for the run.")
    parser.add_argument("--n_cycles", type=int, default=3, help="Number of interpretation-retraining cycles.")
    parser.add_argument("--is_regression", action="store_true", help="Flag indicating regression task.")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Path to the input CSV with columns SMILES,Y,origin(train/val/test).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory.")
    parser.add_argument("--min_value_regression", type=float, default=0.5, help="Minimum property value threshold for selecting molecules (regression only).")
    parser.add_argument("--task", choices=["lipophilicity", "ames"], default="lipophilicity", help="Task for plausibility classification of rationales.")
    parser.add_argument("--max_molecules_for_plausibility", type=int, default=50, help="Max molecules to process for rationales per cycle.")
    parser.add_argument("--use_llm_plausibility", action="store_true", help="Use LLM-based plausibility scoring instead of rule-based.")
    parser.add_argument("--task_description", type=str, default="", help="Short description of the task (context for LLM plausibility).")
    parser.add_argument("--dataset_description", type=str, default="", help="Short dataset description (context for LLM plausibility).")

    parser.add_argument("--total_epochs", type=int, default=200, help="Total training epochs across all cycles (single value).")
    parser.add_argument("--reinterpretations", type=int, default=10, help="Number of interpretation-retraining cycles (single value).")
    parser.add_argument("--learning_rates", type=float, default=1e-3, help="Learning rate for triplet training (single value).")
    parser.add_argument("--triplet_weights", type=float, default=1.0, help="Triplet loss weight (single value).")
    parser.add_argument("--margins", type=float, default=0.1, help="Triplet margin (single value).")
    parser.add_argument("--n_replicates", type=int, default=1, help="Number of replicates for the run (single value).")

    args = parser.parse_args()

    global plausibility_fn
    if args.task == "lipophilicity":
        plausibility_fn = classify_logp_rationales
    elif args.task == "ames":
        plausibility_fn = classify_ames_rationales
  
    run_folder_name = args.run_id
    output_dir = os.path.join(args.output_dir, run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv_path)
    train_df = df[df['origin'] == 'train'].copy()
    val_df = df[df['origin'] == 'val'].copy()
    test_df = df[df['origin'] == 'test'].copy()

    # Use single values for hyperparameters and pass them directly to training
    n_reps = max(1, int(args.n_replicates))
    n_cycles_combo = max(1, int(args.reinterpretations))
    total_epochs = max(1, int(args.total_epochs))
    triplet_epochs = max(1, total_epochs // n_cycles_combo)
    triplet_lr = float(args.learning_rates)
    triplet_weight = float(args.triplet_weights)
    margin = float(args.margins)

    if n_reps == 1:
        live_csv_path = os.path.join(output_dir, 'single_run_results.csv')
        plaus_log_path = os.path.join(output_dir, 'plausibility_ratings_1.csv')
        cycle_results = run_multi_cycle_training(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            n_cycles=n_cycles_combo,
            triplet_epochs=triplet_epochs,
            triplet_lr=triplet_lr,
            triplet_weight=triplet_weight,
            margin=margin,
            property_name='Y',
            output_dir=output_dir,
            combo_idx=1,
            is_regression=args.is_regression,
            min_value_regression=args.min_value_regression,
            max_molecules_for_plausibility=args.max_molecules_for_plausibility,
            use_llm_plausibility=args.use_llm_plausibility,
            task_description=args.task_description,
            dataset_description=args.dataset_description,
            live_results_path=live_csv_path,
            replicate_idx=1,
            plausibility_log_path=plaus_log_path,
        )
        pd.DataFrame(cycle_results).to_csv(os.path.join(output_dir, 'single_run_results.csv'), index=False)
    else:
        for rep in range(1, n_reps + 1):
            live_csv_path = os.path.join(output_dir, f'results_{rep}.csv')
            plaus_log_path = os.path.join(output_dir, f'plausibility_ratings_{rep}.csv')
            run_multi_cycle_training(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                n_cycles=n_cycles_combo,
                triplet_epochs=triplet_epochs,
                triplet_lr=triplet_lr,
                triplet_weight=triplet_weight,
                margin=margin,
                property_name='Y',
                output_dir=output_dir,
                combo_idx=1,
                is_regression=args.is_regression,
                min_value_regression=args.min_value_regression,
                max_molecules_for_plausibility=args.max_molecules_for_plausibility,
                use_llm_plausibility=args.use_llm_plausibility,
                task_description=args.task_description,
                dataset_description=args.dataset_description,
                live_results_path=live_csv_path,
                replicate_idx=rep,
                plausibility_log_path=plaus_log_path,
            )


if __name__ == "__main__":
    # Initialize default plausibility function
    plausibility_fn = classify_logp_rationales
    main()


