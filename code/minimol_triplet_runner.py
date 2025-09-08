import os
import argparse
from typing import Dict, List, Optional

import random
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from minimol import Minimol
import os
import sys
import re
import unicodedata
from rdkit import Chem

BASE = os.path.dirname(os.path.abspath(__file__))        
PARENT = os.path.abspath(os.path.join(BASE, ".."))       
sys.path.insert(0, PARENT)

sys.path.insert(0, BASE)
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Recap
from minimol_triplet_model import (
    train_minimol_triplet,
    MinimolTripletModule,
    MinimolTripletDataset,
    minimol_triplet_collate,
    CachedEncoder,
)
import torch
from torch.utils.data import DataLoader
import math

# Classifiers for different tasks

from plausibility_utils import (
        classify_logp_rationales as classify_lipophilicity_rationales,
        classify_ames_rationales,
)
# Optional LLM-assisted plausibility imports
try:
    from LLM_assisted_plausibility.get_plausibility import get_plausibility  # type: ignore
except Exception:
    get_plausibility = None  # type: ignore
    print('failed to import LLM_assisted_plausibility')

# Optional import placeholders for a second plausibility function per task
try:
    from plausibility_utils import (
        classify_logp_rationales_neg as classify_lipophilicity_rationales_neg,
    )
except Exception:
    classify_lipophilicity_rationales_neg = None

#temp 
classify_ames_rationales_neg = None

#from mcts_minimol import build_plausibility_mcts, get_cleaned_fragments


def _canon(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m else smiles

# (MCTS and fragment utilities are provided by mcts_minimol)
def get_cleaned_fragments(smiles: str) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    fragments = set()
    try:
        fragments.update(BRICS.BRICSDecompose(mol, keepNonLeafNodes=True))
        print('brics')
        print(fragments)
    except Exception:
        print('brics failed')
        pass 

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
        if scaffold_smiles:
            fragments.add(scaffold_smiles)
        print('murcko')
        print(scaffold_smiles)
    except Exception:
        print('murcko failed')
        pass
    try: 

        recap_tree = Recap.RecapDecompose(mol)   # returns a tree
        if recap_tree:
            recap_frags = set(recap_tree.GetAllChildren().keys())
        fragments.update(recap_frags)
        print('recap')

        print(recap_frags)
    except Exception:
        print('recap failed')
        pass
    valid_fragments: List[str] = []
    cleaned_fragments: List[str] = []
    fragments_updated =[]

    for smi in fragments:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            smi_clean = Chem.MolToSmiles(mol,canonical=True)
            fragments_updated.append(smi_clean)
        fragments = fragments_updated
    STAR_LIKE = r"\*\u2217\u22C6\u2731\u204E\uFE61\uFF0A\u2605\u2606"

    BRACKETED_PATTERNS = [
        r"\[\d+[" + STAR_LIKE + r"]\]",   # [14*]
        r"\[\*:\d+\]",                    # [*:14]
        r"\[\d+\]",                       # [14]
        r"\[\*\]",                        # [*]
        r"\(\s*[" + STAR_LIKE + r"]\s*\)",# (*)
        r"\(\)",                          # ()
    ]

    def preclean(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        # strip zero-width stuff
        s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
        for p in BRACKETED_PATTERNS:
            s = re.sub(p, "", s)
        # remove any remaining loose star-like characters
        s = re.sub("[" + STAR_LIKE + "]", "", s)
        return s

    def strip_dummy_atoms_with_rdkit(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return smi
        # remove all atoms with atomic number 0 (dummy '*')
        rw = Chem.RWMol(mol)
        to_del = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        # delete from highest index to lowest
        for idx in sorted(to_del, reverse=True):
            rw.RemoveAtom(idx)
        cleaned = rw.GetMol()
        # re-canonicalize
        return Chem.MolToSmiles(cleaned, canonical=True)

    cleaned_fragments = []
    for frag in fragments:
        s = preclean(frag)
        s = strip_dummy_atoms_with_rdkit(s)
        cleaned_fragments.append(s)

    valid_fragments = []
    for smi in cleaned_fragments:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            n = mol.GetNumAtoms()
            if 5 <= n <= 11:
                valid_fragments.append(smi)

    # Deduplicate by similarity
    filtered: List[str] = []
    selected_fps: List = []
    for smi in valid_fragments:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        is_similar = any(DataStructs.TanimotoSimilarity(fp, prev_fp) >= 0.4 for prev_fp in selected_fps)
        if is_similar:
            continue
        selected_fps.append(fp)
        filtered.append(smi)
    return filtered


def _build_plausibility(
    train_df: pd.DataFrame,
    classifier_fn,
    max_frags_per_mol: Optional[int],
    *,
    is_regression: bool,
    min_score: Optional[float] = None,
) -> tuple[Dict[str, Dict[str, List]], int, int]:
    mapping: Dict[str, Dict[str, List]] = {}
    # Filter candidate molecules for substructure extraction
    if is_regression:
        if min_score is not None:
            cand_df = train_df[train_df['Y'] >= float(min_score)]
        else:
            cand_df = train_df
    else:
        cand_df = train_df[train_df['Y'] == 1]

    uniq_smiles = list(dict.fromkeys([str(s) for s in cand_df['SMILES'].tolist()]))
    entered = 0
    with_pair = 0
    for s in uniq_smiles:
        cands = get_cleaned_fragments(s)
        if len(cands) < 2:
            continue

        entered += 1
        # If there are more than 4 fragments, randomly select 4
        if len(cands) > 5:
            cands = random.sample(cands, 5)
        try:
            labels = classifier_fn(cands)
        except Exception:
            continue
        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1
    return mapping, entered, with_pair


def _build_plausibility_two_sided(
    train_df: pd.DataFrame,
    classifier_pos_fn,
    classifier_neg_fn,
    *,
    is_regression: bool,
    min_score: Optional[float] = None,
    max_neg_value: Optional[float] = None,
) -> tuple[Dict[str, Dict[str, List]], int, int]:
    mapping: Dict[str, Dict[str, List]] = {}
    if is_regression:
        pos_df = train_df if min_score is None else train_df[train_df['Y'] >= float(min_score)]
        if max_neg_value is None:
            neg_df = train_df
        else:
            neg_df = train_df[train_df['Y'] <= float(max_neg_value)]
    else:
        pos_df = train_df[train_df['Y'] == 1]
        neg_df = train_df[train_df['Y'] == 0]

    uniq_smiles_pos = list(dict.fromkeys([str(s) for s in pos_df['SMILES'].tolist()]))
    uniq_smiles_neg = list(dict.fromkeys([str(s) for s in neg_df['SMILES'].tolist()]))

    entered = 0
    with_pair = 0

    # Process positives
    for s in uniq_smiles_pos:
        cands = get_cleaned_fragments(s)
        if len(cands) < 2:
            continue
        entered += 1
        if len(cands) > 5:
            cands = random.sample(cands, 5)
        try:
            labels = classifier_pos_fn(cands)
        except Exception:
            continue
        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1

    # Process negatives
    for s in uniq_smiles_neg:
        cands = get_cleaned_fragments(s)
        if len(cands) < 2:
            continue
        entered += 1
        if len(cands) > 5:
            cands = random.sample(cands, 5)
        try:
            labels = classifier_neg_fn(cands)
        except Exception:
            continue
        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1

    return mapping, entered, with_pair


import json
import tempfile
import subprocess


def set_reproducible_seed(seed: int) -> None:
    """Set seeds across libraries for reproducibility."""
    try:
        from lightning import pytorch as pl  # type: ignore
        pl.seed_everything(int(seed), workers=True)
    except Exception:
        pass
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch  # local import to avoid circulars if any
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
    try:
        os.environ['PYTHONHASHSEED'] = str(int(seed))
    except Exception:
        pass


def _describe_molecules_via_script(smiles_list: List[str], n_workers: int = 20) -> Dict[str, str]:
    """Invoke the external shell script to describe molecules in a separate conda env.

    Returns a mapping {smiles: description}. Missing entries default to ''.
    """
    # Resolve absolute paths
    script_path = os.path.join(BASE, 'run_describe_molecules.sh')
    if not os.path.isfile(script_path):
        # If not found relative to this file, try project root structure
        alt_path = os.path.abspath(os.path.join(PARENT, 'code', 'run_describe_molecules.sh'))
        script_path = alt_path if os.path.isfile(alt_path) else script_path

    # Write temporary inputs/outputs
    with tempfile.TemporaryDirectory() as td:
        smiles_json = os.path.join(td, 'smiles.json')
        output_json = os.path.join(td, 'descriptions.json')
        with open(smiles_json, 'w') as f:
            json.dump(smiles_list, f)

        try:
            cmd = [
                '/bin/bash',
                script_path,
                smiles_json,
                output_json,
                str(int(n_workers)),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            # On any failure, return empty descriptions
            return {s: '' for s in smiles_list}

        # Read output mapping
        try:
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    mapping = json.load(f)
                # Ensure full coverage
                return {s: (mapping.get(s, '') or '') for s in smiles_list}
        except Exception:
            pass
        return {s: '' for s in smiles_list}


def _build_plausibility_llm(
    train_df: pd.DataFrame,
    *,
    is_regression: bool,
    min_score: Optional[float] = None,
    task_description: str = '',
    dataset_description: str = '',
    max_llm_molecules: Optional[int] = None,
) -> tuple[Dict[str, Dict[str, List]], int, int]:
    if get_plausibility is None:
        raise RuntimeError("LLM-assisted plausibility requested but LLM scoring module is not available.")

    mapping: Dict[str, Dict[str, List]] = {}
    # Candidate parent molecules
    if is_regression:
        cand_df = train_df if min_score is None else train_df[train_df['Y'] >= float(min_score)]
    else:
        cand_df = train_df[train_df['Y'] == 1]

    uniq_smiles = list(dict.fromkeys([str(s) for s in cand_df['SMILES'].tolist()]))
    # Filter to parents that have at least 2 fragments first
    eligible: List[tuple[str, List[str]]] = []
    for s in uniq_smiles:
        cs = get_cleaned_fragments(s)
        if len(cs) >= 2:
            eligible.append((s, cs))

    # Then, enforce the max cap over eligible parents
    if isinstance(max_llm_molecules, int) and max_llm_molecules is not None and max_llm_molecules > 0 and len(eligible) > max_llm_molecules:
        eligible = random.sample(eligible, int(max_llm_molecules))

    entered = 0
    with_pair = 0

    for s, cands in eligible:
        entered += 1
        # Cap at 5 using random sampling when more are available
        if len(cands) > 5:
            cands = random.sample(cands, 5)

        # Describe parent and candidate fragments via external script in separate env
        all_smiles = cands + [s]
        print(all_smiles)
        descriptions = _describe_molecules_via_script(all_smiles)

        desc_list = [descriptions.get(r, '') for r in cands]
        original_desc = descriptions.get(s, '')
        print(desc_list)
        print('getting llm plausibility')
        try:
            labels = get_plausibility(
                description_list=desc_list,
                smiles_list=cands,
                original_smiles=s,
                original_molecule_description=original_desc,
                task_description=task_description,
                dataset_description=dataset_description,
            )
            print(labels)
        except Exception:
            labels = [0] * len(cands)
            print(labels)

        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1

    return mapping, entered, with_pair


def _build_plausibility_llm_two_sided(
    train_df: pd.DataFrame,
    *,
    is_regression: bool,
    min_score: Optional[float] = None,
    max_neg_value: Optional[float] = None,
    task_description_pos: str = '',
    task_description_neg: str = '',
    dataset_description: str = '',
    max_llm_molecules: Optional[int] = None,
) -> tuple[Dict[str, Dict[str, List]], int, int]:
    if get_plausibility is None:
        raise RuntimeError("LLM-assisted plausibility requested but LLM scoring module is not available.")

    mapping: Dict[str, Dict[str, List]] = {}

    if is_regression:
        pos_df = train_df if min_score is None else train_df[train_df['Y'] >= float(min_score)]
        if max_neg_value is None:
            neg_df = train_df
        else:
            neg_df = train_df[train_df['Y'] <= float(max_neg_value)]
    else:
        pos_df = train_df[train_df['Y'] == 1]
        neg_df = train_df[train_df['Y'] == 0]

    uniq_smiles_pos = list(dict.fromkeys([str(s) for s in pos_df['SMILES'].tolist()]))
    uniq_smiles_neg = list(dict.fromkeys([str(s) for s in neg_df['SMILES'].tolist()]))

    # Build eligible lists (require at least 2 fragments)
    eligible_pos: List[tuple[str, List[str]]] = []
    for s in uniq_smiles_pos:
        cs = get_cleaned_fragments(s)
        if len(cs) >= 2:
            eligible_pos.append((s, cs))

    eligible_neg: List[tuple[str, List[str]]] = []
    for s in uniq_smiles_neg:
        cs = get_cleaned_fragments(s)
        if len(cs) >= 2:
            eligible_neg.append((s, cs))

    # Apply optional caps per branch
    if isinstance(max_llm_molecules, int) and max_llm_molecules is not None and max_llm_molecules > 0:
        if len(eligible_pos) > max_llm_molecules:
            eligible_pos = random.sample(eligible_pos, int(max_llm_molecules))
        if len(eligible_neg) > max_llm_molecules:
            eligible_neg = random.sample(eligible_neg, int(max_llm_molecules))

    entered = 0
    with_pair = 0

    # Process positives with positive task description
    for s, cands in eligible_pos:
        entered += 1
        if len(cands) > 5:
            cands = random.sample(cands, 5)
        all_smiles = cands + [s]
        print(all_smiles)
        descriptions = _describe_molecules_via_script(all_smiles)
        desc_list = [descriptions.get(r, '') for r in cands]
        print(desc_list)
        original_desc = descriptions.get(s, '')
        try:
            labels = get_plausibility(
                description_list=desc_list,
                smiles_list=cands,
                original_smiles=s,
                original_molecule_description=original_desc,
                task_description=task_description_pos,
                dataset_description=dataset_description,
            )
        except Exception:
            labels = [0] * len(cands)

        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1

    # Process negatives with negative task description
    for s, cands in eligible_neg:
        entered += 1
        if len(cands) > 5:
            cands = random.sample(cands, 5)
        all_smiles = cands + [s]
        descriptions = _describe_molecules_via_script(all_smiles)
        desc_list = [descriptions.get(r, '') for r in cands]
        original_desc = descriptions.get(s, '')
        try:
            labels = get_plausibility(
                description_list=desc_list,
                smiles_list=cands,
                original_smiles=s,
                original_molecule_description=original_desc,
                task_description=task_description_neg,
                dataset_description=dataset_description,
            )
        except Exception:
            labels = [0] * len(cands)

        if 1 in labels and -1 in labels:
            mapping[s] = {'Rationales': cands, 'plausibility': labels}
            with_pair += 1

    return mapping, entered, with_pair

def _evaluate_regression_module(model: MinimolTripletModule, df: pd.DataFrame, encoder: CachedEncoder, batch_size: int = 256):
    model.eval()
    ds = MinimolTripletDataset(df, smiles_to_plausibility=None, require_triplet=False)
    collate = minimol_triplet_collate(encoder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    yp, yt = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b['x'], b['y']
            p = model(x).detach().cpu().numpy()
            yp.extend(p.tolist())
            yt.extend(y.detach().cpu().numpy().tolist())
    yp = np.array(yp)
    yt = np.array(yt)
    rmse = float(np.sqrt(((yp - yt) ** 2).mean()))
    mae = float(np.abs(yp - yt).mean())
    denom = float(((yt - yt.mean()) ** 2).sum())
    r2 = float(1.0 - (((yp - yt) ** 2).sum() / denom)) if denom > 0 else 0.0
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def _evaluate_classification_module(model: MinimolTripletModule, df: pd.DataFrame, encoder: CachedEncoder, batch_size: int = 256):
    model.eval()
    ds = MinimolTripletDataset(df, smiles_to_plausibility=None, require_triplet=False)
    collate = minimol_triplet_collate(encoder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    yp_raw, yt = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b['x'], b['y']
            p = model(x).detach().cpu().numpy()
            yp_raw.extend(p.tolist())
            yt.extend(y.detach().cpu().numpy().tolist())
    yp_raw = np.array(yp_raw)
    yt = np.array(yt)
    # probabilities
    yp_prob = 1.0 / (1.0 + np.exp(-yp_raw))
    auroc = None
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        # Only compute AUROC if both classes present
        if np.unique(yt).size == 2:
            auroc = float(roc_auc_score(yt, yp_prob))
    except Exception:
        auroc = None
    return {'auroc': auroc}


def run_minimol_triplet(
    csv_path: str,
    output_dir: str,
    cache_file: Optional[str],
    lrs: List[float],
    epochs_list: List[int],
    repr_sizes_list: List[int],
    hidden_sizes_list: List[int],
    depths_list: List[int],
    batch_size: int,
    margins: List[float],
    triplet_weights: List[float],
    max_frags_per_mol: Optional[int],
    replicates: int,
    task: str,
    min_score: Optional[float] = None,
    get_two_sided: bool = False,
    max_neg_value: Optional[float] = None,
    combine: bool = False,
    combines: Optional[List[bool]] = None,
    use_batch_norm: bool = True,
    bn_flags: Optional[List[bool]] = None,
    use_mcts: bool = False,
    mcts_rollout: int = 5,
    mcts_c_puct: float = 75.0,
    mcts_max_atoms: int = 11,
    mcts_min_atoms: int = 5,
    mcts_prop_delta: float = -10.0,
    use_llm_plausibility: bool = False,
    task_description: str = '',
    task_description_negative: str = '',
    dataset_description: str = '',
    max_llm_molecules: Optional[int] = None,
    is_regression: bool = False,
    dropout: float = 0.1,
):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f'grid_results_{task}.csv')
    df = pd.read_csv(csv_path)
    if 'SMILES_CANON' not in df.columns:
        df['SMILES_CANON'] = df['SMILES'].apply(_canon)

    train_df = df[df['origin'] == 'train'].copy()
    val_df = df[df['origin'] == 'val'].copy()
    test_df = df[df['origin'] == 'test'].copy()

    # Select classifier based on task (independent of regression/classification training mode)
    task_lc = task.lower()
    if task_lc == 'lipophilicity':
        classifier_fn = classify_lipophilicity_rationales
        classifier_alt_fn = classify_lipophilicity_rationales_neg

    elif task_lc == 'ames':
        classifier_fn = classify_ames_rationales
        classifier_alt_fn = classify_ames_rationales_neg
    else:
        print("task is", task)
        if use_llm_plausibility == False:
            raise ValueError(f"Unknown task: {task}. Expected one of: lipophilicity, solubility, ames")

    # Minimol encoder + cache
    print(
        f"Initializing Minimol encoder and cache..."
    )
    mini = Minimol()
    cache_path = cache_file if cache_file else os.path.join(output_dir, 'smiles_fp_cache.pt')
    encoder = CachedEncoder(mini, cache_path)

    # Build plausibility map
    if bool(use_llm_plausibility) and bool(get_two_sided):
        smiles_to_plaus, n_entered, n_with_pair = _build_plausibility_llm_two_sided(
            train_df=train_df,
            is_regression=is_regression,
            min_score=min_score,
            max_neg_value=max_neg_value,
            task_description_pos=task_description,
            task_description_neg=task_description_negative,
            dataset_description=dataset_description,
            max_llm_molecules=max_llm_molecules,
        )
    elif bool(use_llm_plausibility):
        smiles_to_plaus, n_entered, n_with_pair = _build_plausibility_llm(
            train_df=train_df,
            is_regression=is_regression,
            min_score=min_score,
            task_description=task_description,
            dataset_description=dataset_description,
            max_llm_molecules=max_llm_molecules,
        )
    elif get_two_sided:
        if classifier_alt_fn is None:
            raise ValueError("Two-sided plausibility requested but alternate classifier is not available for this task. Please add it to plausibility_utils.")
        smiles_to_plaus, n_entered, n_with_pair = _build_plausibility_two_sided(
            train_df=train_df,
            classifier_pos_fn=classifier_fn,
            classifier_neg_fn=classifier_alt_fn,
            is_regression=is_regression,
            min_score=min_score,
            max_neg_value=max_neg_value,
        )
    else:
        smiles_to_plaus, n_entered, n_with_pair = _build_plausibility(
            train_df,
            classifier_fn=classifier_fn,
            max_frags_per_mol=max_frags_per_mol,
            is_regression=is_regression,
            min_score=min_score,
        )



    print(
        f"[{task}] Rationale search stats: entered={n_entered}, "
        f"with_plausible_and_implausible_pair={n_with_pair}"
    )

    # Save plausibility mapping to CSV (parent SMILES, rationale fragment, label)
    plaus_rows = []
    for s, entry in smiles_to_plaus.items():
        rats = entry.get('Rationales', [])
        labs = entry.get('plausibility', [])
        for r, l in zip(rats, labs):
            plaus_rows.append({'SMILES': s, 'rationale': r, 'plausibility': int(l)})
    if len(plaus_rows) > 0:
        plaus_out_csv = os.path.join(output_dir, f'rationale_search_results_{task}.csv')
        pd.DataFrame(plaus_rows).to_csv(plaus_out_csv, index=False)
        print(f"Saved plausibility results to {plaus_out_csv}")

    run_idx = 0
    combine_values = combines if (combines is not None and len(combines) > 0) else [bool(combine)]
    bn_values = bn_flags if (bn_flags is not None and len(bn_flags) > 0) else [bool(use_batch_norm)]
    for rep in range(int(max(1, replicates))):
        # Set deterministic seed per replicate: 1,2,3,...
        set_reproducible_seed(int(rep) + 1)
    
        for margin in margins:
            for triplet_weight in triplet_weights:
                for lr in lrs:
                    for n_epochs in epochs_list:
                        for repr_size in repr_sizes_list:
                            for hidden_size in hidden_sizes_list:
                                for depth in depths_list:
                                    if int(depth) < 2:
                                        continue
                                    if depth not in [3, 4]:
                                        continue
                                    for combine_flag in combine_values:
                                        for bn_flag in bn_values:
                                            # 1) Triplet with rule-based plausibility
                                            model = train_minimol_triplet(
                                                train_df=train_df,
                                                val_df=val_df,
                                                encoder=encoder,
                                                is_regression=is_regression,
                                                margin=margin,
                                                triplet_weight=triplet_weight,
                                                init_lr=lr,
                                                max_epochs=n_epochs,
                                                smiles_to_plausibility=smiles_to_plaus,
                                                batch_size=batch_size,
                                                require_triplet_for_train=False,
                                                repr_dim=int(repr_size),
                                                hidden_dim=int(hidden_size),
                                                depth=int(depth),
                                                combine=bool(combine_flag),
                                                use_batch_norm=bool(bn_flag),
                                                dropout=float(dropout),
                                            )

                                            if is_regression:
                                                val_metrics = _evaluate_regression_module(model, val_df, encoder, batch_size=256)
                                                test_metrics = _evaluate_regression_module(model, test_df, encoder, batch_size=256)
                                            else:
                                                val_metrics = _evaluate_classification_module(model, val_df, encoder, batch_size=256)
                                                test_metrics = _evaluate_classification_module(model, test_df, encoder, batch_size=256)

                                            print(
                                            f"Run {run_idx} (rep {rep+1}): lr={lr}, epochs={n_epochs}, margin={margin}, tw={triplet_weight}, repr={repr_size}, hidden={hidden_size}, depth={depth}, combine={combine_flag}, bn={bn_flag} | "
                                            + (f"Val RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, Test RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}" if is_regression else f"Val AUROC={(val_metrics['auroc'] if val_metrics['auroc'] is not None else float('nan')):.4f}, Test AUROC={(test_metrics['auroc'] if test_metrics['auroc'] is not None else float('nan')):.4f}")
                                            )

                                            # 2) No-triplet baseline (triplet_weight=0) using only main loss
                                            model_no_trip = train_minimol_triplet(
                                                train_df=train_df,
                                                val_df=val_df,
                                                encoder=encoder,
                                                is_regression=is_regression,
                                                margin=margin,
                                                triplet_weight=0.0,
                                                init_lr=lr,
                                                max_epochs=n_epochs,
                                                smiles_to_plausibility=None,
                                                batch_size=batch_size,
                                                require_triplet_for_train=False,
                                                repr_dim=int(repr_size),
                                                hidden_dim=int(hidden_size),
                                                depth=int(depth),
                                                combine=bool(combine_flag),
                                                use_batch_norm=bool(bn_flag),
                                                dropout=float(dropout),
                                            )
                                            if is_regression:
                                                val_metrics_nt = _evaluate_regression_module(model_no_trip, val_df, encoder, batch_size=256)
                                                test_metrics_nt = _evaluate_regression_module(model_no_trip, test_df, encoder, batch_size=256)
                                            else:
                                                val_metrics_nt = _evaluate_classification_module(model_no_trip, val_df, encoder, batch_size=256)
                                                test_metrics_nt = _evaluate_classification_module(model_no_trip, test_df, encoder, batch_size=256)
                                            print(
                                            f"Run {run_idx} (rep {rep+1}, no_triplet): lr={lr}, epochs={n_epochs}, margin={margin}, repr={repr_size}, hidden={hidden_size}, depth={depth}, combine={combine_flag}, bn={bn_flag} | "
                                            + (f"Val RMSE={val_metrics_nt['rmse']:.4f}, MAE={val_metrics_nt['mae']:.4f}, Test RMSE={test_metrics_nt['rmse']:.4f}, MAE={test_metrics_nt['mae']:.4f}" if is_regression else f"Val AUROC={(val_metrics_nt['auroc'] if val_metrics_nt['auroc'] is not None else float('nan')):.4f}, Test AUROC={(test_metrics_nt['auroc'] if test_metrics_nt['auroc'] is not None else float('nan')):.4f}")
                                            )

                                            # 3) Random-plausibility baseline: assign one +1 and one -1 per molecule randomly
                                            smiles_to_plaus_random: Dict[str, Dict[str, List]] = {}
                                            for s in list(dict.fromkeys(train_df['SMILES'].tolist())):
                                                cands = get_cleaned_fragments(s)
                                                if len(cands) < 2:
                                                    continue
                                                labels = [0] * len(cands)
                                                i_pos, i_neg = random.sample(range(len(cands)), 2)
                                                labels[i_pos] = 1
                                                labels[i_neg] = -1
                                                smiles_to_plaus_random[s] = {'Rationales': cands, 'plausibility': labels}

                                            model_rand = train_minimol_triplet(
                                                train_df=train_df,
                                                val_df=val_df,
                                                encoder=encoder,
                                                is_regression=is_regression,
                                                margin=margin,
                                                triplet_weight=triplet_weight,
                                                init_lr=lr,
                                                max_epochs=n_epochs,
                                                smiles_to_plausibility=smiles_to_plaus_random,
                                                batch_size=batch_size,
                                                require_triplet_for_train=False,
                                                repr_dim=int(repr_size),
                                                hidden_dim=int(hidden_size),
                                                depth=int(depth),
                                                combine=bool(combine_flag),
                                                use_batch_norm=bool(bn_flag),
                                                dropout=float(dropout),
                                            )
                                            if is_regression:
                                                val_metrics_rd = _evaluate_regression_module(model_rand, val_df, encoder, batch_size=256)
                                                test_metrics_rd = _evaluate_regression_module(model_rand, test_df, encoder, batch_size=256)
                                            else:
                                                val_metrics_rd = _evaluate_classification_module(model_rand, val_df, encoder, batch_size=256)
                                                test_metrics_rd = _evaluate_classification_module(model_rand, test_df, encoder, batch_size=256)
                                            print(
                                            f"Run {run_idx} (rep {rep+1}, random): lr={lr}, epochs={n_epochs}, margin={margin}, tw={triplet_weight}, repr={repr_size}, hidden={hidden_size}, depth={depth}, combine={combine_flag}, bn={bn_flag} | "
                                            + (f"Val RMSE={val_metrics_rd['rmse']:.4f}, MAE={val_metrics_rd['mae']:.4f}, Test RMSE={test_metrics_rd['rmse']:.4f}, MAE={test_metrics_rd['mae']:.4f}" if is_regression else f"Val AUROC={(val_metrics_rd['auroc'] if val_metrics_rd['auroc'] is not None else float('nan')):.4f}, Test AUROC={(test_metrics_rd['auroc'] if test_metrics_rd['auroc'] is not None else float('nan')):.4f}")
                                            )

                                            # Append a single combined row to CSV incrementally
                                            combined_row = {
                                                'run': run_idx,
                                                'replicate': rep + 1,
                                                'lr': lr,
                                                'epochs': n_epochs,
                                                'margin': margin,
                                                'triplet_weight': triplet_weight,
                                                'repr_size': int(repr_size),
                                                'hidden_size': int(hidden_size),
                                                'depth': int(depth),
                                                'combine': bool(combine_flag),
                                                'use_batch_norm': bool(bn_flag),
                                                'n_entered': n_entered,
                                                'n_with_pair': n_with_pair,
                                                # triplet with plausibility
                                                'val_rmse': (val_metrics['rmse'] if is_regression else None),
                                                'val_mae': (val_metrics['mae'] if is_regression else None),
                                                'val_r2': (val_metrics['r2'] if is_regression else None),
                                                'test_rmse': (test_metrics['rmse'] if is_regression else None),
                                                'test_mae': (test_metrics['mae'] if is_regression else None),
                                                'test_r2': (test_metrics['r2'] if is_regression else None),
                                                'val_auroc': (val_metrics['auroc'] if not is_regression else None),
                                                'test_auroc': (test_metrics['auroc'] if not is_regression else None),
                                                'val_acc': None,
                                                'test_acc': None,
                                                # no-triplet baseline
                                                'val_rmse_no_triplet': (val_metrics_nt['rmse'] if is_regression else None),
                                                'val_mae_no_triplet': (val_metrics_nt['mae'] if is_regression else None),
                                                'val_r2_no_triplet': (val_metrics_nt['r2'] if is_regression else None),
                                                'test_rmse_no_triplet': (test_metrics_nt['rmse'] if is_regression else None),
                                                'test_mae_no_triplet': (test_metrics_nt['mae'] if is_regression else None),
                                                'test_r2_no_triplet': (test_metrics_nt['r2'] if is_regression else None),
                                                'val_auroc_no_triplet': (val_metrics_nt['auroc'] if not is_regression else None),
                                                'test_auroc_no_triplet': (test_metrics_nt['auroc'] if not is_regression else None),
                                                'val_acc_no_triplet': None,
                                                'test_acc_no_triplet': None,
                                                # random-plausibility baseline
                                                'val_rmse_random': (val_metrics_rd['rmse'] if is_regression else None),
                                                'val_mae_random': (val_metrics_rd['mae'] if is_regression else None),
                                                'val_r2_random': (val_metrics_rd['r2'] if is_regression else None),
                                                'test_rmse_random': (test_metrics_rd['rmse'] if is_regression else None),
                                                'test_mae_random': (test_metrics_rd['mae'] if is_regression else None),
                                                'test_r2_random': (test_metrics_rd['r2'] if is_regression else None),
                                                'val_auroc_random': (val_metrics_rd['auroc'] if not is_regression else None),
                                                'test_auroc_random': (test_metrics_rd['auroc'] if not is_regression else None),
                                                'val_acc_random': None,
                                                'test_acc_random': None,
                                            }
                                            pd.DataFrame([combined_row]).to_csv(
                                                out_csv,
                                                mode='a',
                                                header=not os.path.exists(out_csv),
                                                index=False,
                                            )

    print(f"Results are being incrementally saved to {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--cache_file', type=str, default=None)
    ap.add_argument('--lrs', type=str, default='3e-4')
    ap.add_argument('--epochs', type=str, default='25')
    ap.add_argument('--repr_sizes', type=str, default='512', help='Comma-separated representation layer sizes to sweep')
    ap.add_argument('--hidden_sizes', type=str, default='512', help='Comma-separated hidden sizes to sweep')
    ap.add_argument('--depths', type=str, default='3', help='Comma-separated depths to sweep (min 2; allowed 3 or 4)')
    ap.add_argument('--combine', action='store_true', help='Concatenate original Minimol embedding with final hidden for prediction')
    ap.add_argument('--combines', type=str, default=None, help='Comma-separated booleans to sweep combine flag, e.g., true,false')
    ap.add_argument('--no_batch_norm', action='store_true', help='Disable BatchNorm layers in the MLP')
    ap.add_argument('--bn_flags', type=str, default=None, help='Comma-separated booleans to sweep BN on/off, e.g., true,false')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--margin', type=float, default=0.2)
    ap.add_argument('--triplet_weight', type=float, default=1.0)
    ap.add_argument('--margins', type=str, default=None, help='Comma-separated margins to sweep')
    ap.add_argument('--triplet_weights', type=str, default=None, help='Comma-separated triplet weights to sweep')
    ap.add_argument('--dropout', type=float, default=0.1, help='Dropout probability for MLP layers (0.0-1.0)')
    ap.add_argument('--max_frags_per_mol', type=int, default=8)
    ap.add_argument('--replicates', type=int, default=1)
    ap.add_argument('--task', type=str, default='lipophilicity')
    ap.add_argument('--min_score', type=float, default=None, help='Regression only: minimum Y to include for plausibility extraction')
    # Two-sided plausibility options
    ap.add_argument('--get_two_sided', action='store_true', help='Use two plausibility functions: one for hits and one for non-hits')
    ap.add_argument('--max_neg_value', type=float, default=None, help='Regression only: max Y value to consider for negative branch when two-sided')
    # LLM plausibility options
    ap.add_argument('--use_llm_plausibility', action='store_true', help='Use LLM-assisted plausibility scoring of fragments')
    ap.add_argument('--task_description', type=str, default='', help='Short description of the task for LLM context')
    ap.add_argument('--task_description_negative', type=str, default='', help='Short description for the negative branch when using two-sided LLM plausibility')
    ap.add_argument('--dataset_description', type=str, default='', help='Short dataset description for LLM context')
    ap.add_argument('--max_llm_molecules', type=int, default=None, help='Max number of parent molecules to evaluate with LLM (randomly sampled if exceeded)')
    ap.add_argument('--is_regression', action='store_true', help='If set, train/evaluate as regression; otherwise classification')
    args = ap.parse_args()
    lrs = [float(x.strip()) for x in args.lrs.split(',')]
    epochs = [int(x.strip()) for x in args.epochs.split(',')]
    if args.margins is not None and args.margins.strip() != '':
        margins = [float(x.strip()) for x in args.margins.split(',')]
    else:
        margins = [float(args.margin)]
    if args.triplet_weights is not None and args.triplet_weights.strip() != '':
        triplet_weights = [float(x.strip()) for x in args.triplet_weights.split(',')]
    else:
        triplet_weights = [float(args.triplet_weight)]
    repr_sizes = [int(x.strip()) for x in args.repr_sizes.split(',')]
    hidden_sizes = [int(x.strip()) for x in args.hidden_sizes.split(',')]
    depths = [int(x.strip()) for x in args.depths.split(',')]
    return args, lrs, epochs, margins, triplet_weights, repr_sizes, hidden_sizes, depths


if __name__ == '__main__':
    args, lrs, epochs_list, margins, triplet_weights, repr_sizes_list, hidden_sizes_list, depths_list = parse_args()
    # Parse combines list (optional)
    combines_list = None
    if args.combines is not None and str(args.combines).strip() != '':
        def _to_bool(s: str) -> bool:
            v = s.strip().lower()
            return v in ('1', 'true', 't', 'yes', 'y')
        combines_list = [_to_bool(s) for s in str(args.combines).split(',')]
    # Parse bn_flags list (optional)
    bn_flags_list = None
    if args.bn_flags is not None and str(args.bn_flags).strip() != '':
        def _to_bool(s: str) -> bool:
            v = s.strip().lower()
            return v in ('1', 'true', 't', 'yes', 'y')
        bn_flags_list = [_to_bool(s) for s in str(args.bn_flags).split(',')]

    run_minimol_triplet(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
        lrs=lrs,
        epochs_list=epochs_list,
        repr_sizes_list=repr_sizes_list,
        hidden_sizes_list=hidden_sizes_list,
        depths_list=depths_list,
        batch_size=args.batch_size,
        margins=margins,
        triplet_weights=triplet_weights,
        max_frags_per_mol=args.max_frags_per_mol,
        replicates=args.replicates,
        task=args.task,
        min_score=args.min_score,
        get_two_sided=args.get_two_sided,
        max_neg_value=args.max_neg_value,
        combine=args.combine,
        combines=combines_list,
        use_batch_norm=(not args.no_batch_norm),
        bn_flags=bn_flags_list,
        use_llm_plausibility=bool(args.use_llm_plausibility),
        task_description=str(args.task_description),
        task_description_negative=str(args.task_description_negative),
        dataset_description=str(args.dataset_description),
        max_llm_molecules=args.max_llm_molecules,
        is_regression=bool(args.is_regression),
        dropout=float(args.dropout),
   
    )


