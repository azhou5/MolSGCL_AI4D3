# for regression

import pandas as pd
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

# -------------------
# Config
# -------------------
input_path = # TODO INSERT .tab file from TDC. 
output_path = # TODO INSERT PATH 

target_train_size = 150        
val_frac_of_test = 0.20       
random_seed = 42
sample_size_for_scoring = 8000  

random.seed(random_seed)
np.random.seed(random_seed)

# -------------------
# Load & featurize
# -------------------
df = pd.read_csv(input_path, sep='\t')

def mol_and_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return mol, fp

df[['mol', 'fp']] = df['Drug'].apply(lambda s: pd.Series(mol_and_fp(s)))
df = df[df['fp'].notnull()].reset_index(drop=True)

# -------------------
# Target check (regression)
# -------------------
if 'Y' not in df.columns:
    raise KeyError("Expected a column 'Y' for the regression target.")
df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
df = df[df['Y'].notnull()].reset_index(drop=True)

# Convenience
fps = list(df['fp'])
N = len(fps)
all_idx = list(range(N))

# -------------------
# Helpers for similarity scoring
# -------------------
def max_sim_to_pool(fp_probe, pool_fps):
    """Maximum Tanimoto similarity of a fingerprint to a pool of fps."""
    if not pool_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp_probe, pool_fps)
    return float(np.max(sims)) if sims else 0.0

def mean_sim_to_pool(fp_probe, pool_fps):
    """Mean Tanimoto similarity of a fingerprint to a pool of fps (tie-breaker)."""
    if not pool_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp_probe, pool_fps)
    return float(np.mean(sims)) if sims else 0.0

# -------------------
# Build TRAIN to minimize similarity with the future TEST set
# Greedy: at each step, among remaining candidates, choose the one
# with the *lowest MAX similarity* to the current remaining pool (future TEST).
# Secondary tie-breaker: lowest MEAN similarity to the pool.
# We sample the pool for speed when it's large.
# -------------------
def build_train_min_test_sim(target_k, sample_size=512):
    remaining = set(all_idx)
    selected = []
    pbar = tqdm(total=min(target_k, N), desc="Selecting TRAIN (min similarity to TEST)")

    while len(selected) < min(target_k, N) and remaining:
        rem_list = list(remaining)

        # Sample from remaining for scoring speed; if small, use all
        if len(rem_list) <= sample_size:
            pool_idx = rem_list
        else:
            pool_idx = random.sample(rem_list, sample_size)
        pool_fps_full = [fps[i] for i in pool_idx]

        best_idx = None
        best_max = float('inf')
        best_mean = float('inf')

        for i in rem_list:
            # Avoid trivial self-similarity if i is in the sampled pool
            if i in pool_idx and len(pool_idx) > 1:
                pool_fps = [fps[j] for j in pool_idx if j != i]
            else:
                pool_fps = pool_fps_full

            # Primary objective: minimize max sim to (future) TEST
            cand_max = max_sim_to_pool(fps[i], pool_fps)

            # Secondary objective: minimize mean sim to (future) TEST
            cand_mean = mean_sim_to_pool(fps[i], pool_fps)

            if (cand_max < best_max) or (cand_max == best_max and cand_mean < best_mean):
                best_max = cand_max
                best_mean = cand_mean
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)
        pbar.update(1)

    pbar.close()
    return selected, remaining

subset_train, remainder = build_train_min_test_sim(target_train_size, sample_size=sample_size_for_scoring)

# -------------------
# Assign splits:
# - TRAIN: selected by min similarity to future TEST
# - VAL: sampled from TEST
# - TEST: all remaining
# -------------------
df_train = df.loc[subset_train].reset_index(drop=True)
df_test  = df.loc[list(remainder)].reset_index(drop=True)

df_train['origin'] = 'train'
df_test['origin']  = 'test'

# Sample VAL **from TEST**
n_val = int(np.floor(val_frac_of_test * len(df_test)))
if n_val > 0 and len(df_test) > 0:
    val_indices = np.random.choice(df_test.index, size=n_val, replace=False)
    df_test.loc[val_indices, 'origin'] = 'val'

# Combine back
combined = pd.concat([df_train, df_test], ignore_index=True)

# Clean columns for downstream usage
combined = combined.drop(columns=['mol', 'fp'], errors='ignore')
combined = combined.rename(columns={'Drug': 'SMILES'})
combined = combined.drop(columns=['Drug_ID'], errors='ignore')

# Save
combined.to_csv(output_path, index=False)

# -------------------
# Summaries (regression)
# -------------------
def summarize_split_reg(frame):
    n = int(len(frame))
    if n == 0:
        return {'n': 0}
    y = frame['Y'].astype(float).values
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    return {
        'n': n,
        'mean': float(np.mean(y)),
        'std': float(np.std(y, ddof=1)) if n > 1 else float('nan'),
        'min': float(np.min(y)),
        'q25': float(q25),
        'median': float(q50),
        'q75': float(q75),
        'max': float(np.max(y)),
    }

# -------------------
# Train–Test similarity metrics
# -------------------
# Rebuild fps for splits from original df
mask_train = df.index.isin(subset_train)
mask_test  = df.index.isin(list(remainder))
train_fps = list(df.loc[mask_train, 'fp'])
test_fps  = list(df.loc[mask_test, 'fp'])

def mean_cross_tanimoto(train_fps, test_fps):
    """Mean pairwise Tanimoto across all train–test pairs."""
    if len(train_fps) == 0 or len(test_fps) == 0:
        return float('nan')
    total = 0.0
    count = 0
    for fp_tr in train_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp_tr, test_fps)
        total += float(np.sum(sims))
        count += len(sims)
    return total / count if count else float('nan')

def mean_max_tanimoto(train_fps, test_fps):
    """Mean over train of (max similarity to any test)."""
    if len(train_fps) == 0 or len(test_fps) == 0:
        return float('nan')
    max_sims = []
    for fp_tr in train_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp_tr, test_fps)
        if sims:
            max_sims.append(max(sims))
    return float(np.mean(max_sims)) if max_sims else float('nan')

mean_cross_sim = mean_cross_tanimoto(train_fps, test_fps)
mean_max_sim   = mean_max_tanimoto(train_fps, test_fps)

# -------------------
# Print report
# -------------------
print("\nSuccess with TEST-overlap minimization (max-sim objective).")
print("Subset sizes:", {k:int(v) for k,v in combined['origin'].value_counts().to_dict().items()})

print("\nRegression target (Y) summary by split:")
for split in ['train','val','test']:
    stats = summarize_split_reg(combined[combined['origin']==split])
    print(f"{split}: {stats}")

print(f"\nMean pairwise Tanimoto similarity BETWEEN TRAIN and TEST: {mean_cross_sim:.6f}")
print(f"Mean MAX Tanimoto similarity (each TRAIN vs most similar TEST): {mean_max_sim:.6f}")
print("\nNote: TRAIN selection minimized each candidate’s *max* similarity to the future TEST pool at selection time, directly reducing worst-case train–test overlap.")
