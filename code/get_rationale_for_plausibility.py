import pandas as pd
import torch
from rdkit import Chem
#import chemprop.data as data
import os
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import math
from typing import Callable, Union, Iterable, List, Tuple, Set, Dict
from dataclasses import dataclass, field
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem # Needed for fingerprint generators
from rdkit import DataStructs
import sys
import chemprop_custom.data as data

from chemprop_custom.models.utils import save_model, load_model
from chemprop_custom.models.utils import save_model, load_model
from chemprop_custom.data import MoleculeDatapoint
from chemprop_custom.utils import make_mol

from chemprop_custom.models.model import MPNN
from chemprop_custom.models.multi import MulticomponentMPNN
from chemprop_custom import nn
from chemprop_custom import featurizers
import io
from chemprop_custom.nn import BondMessagePassing,MeanAggregation, BinaryClassificationFFN
import argparse
import multiprocessing
from functools import partial
from tqdm import tqdm
import time

try:
    multiprocessing.set_start_method('forkserver')
except RuntimeError:
    pass

# This finds rationales for molecules based on MCTS. This is adapted from chemprop 

### params for rationale search

rollout = 5 # number of MCTS rollouts to perform. 
c_puct = 75.0  # constant that controls the level of exploration
max_atoms = 14 # maximum number of atoms allowed in an extracted rationale
min_atoms = 5 # minimum number of atoms in an extracted rationale
prop_delta = 0 # Minimum score to count as positive.
num_rationales_to_keep = 20 

# In this algorithm, if the predicted property from the substructure if larger than prop_delta, the substructure is considered satisfactory
### Load the model (Keep this in the main scope for the single-process case)

def load_mpnn_model(path, map_location):
    """Helper function to load the model."""
    return MPNN.load_from_file(path, map_location)


#functions/ classes
@dataclass
class MCTSNode:
    """Represents a node in a Monte Carlo Tree Search.

    Parameters
    ----------
    smiles : str
        The SMILES for the substructure at this node.
    atoms : list
        A list of atom indices in the substructure at this node.
    W : float
        The total action value, which indicates how likely the deletion will lead to a good rationale.
    N : int
        The visit count, which indicates how many times this node has been visited. It is used to balance exploration and exploitation.
    P : float
        The predicted property score of the new subgraphs' after the deletion, shown as R in the original paper.
    """

    smiles: str
    atoms: Iterable[int]
    W: float = 0
    N: int = 0
    P: float = 0
    children: List['MCTSNode'] = field(default_factory=list)


    def __post_init__(self):
        self.atoms = set(self.atoms)

    def Q(self) -> float:
        """
        Returns
        -------
        float
            The mean action value of the node.
        """
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int, c_puct: float = 10.0) -> float:
        """
        Parameters
        ----------
        n : int
            The sum of the visit count of this node's siblings.
        c_puct : float
            A constant that controls the level of exploration.
        
        Returns
        -------
        float
            The exploration value of the node.
        """
        return c_puct * self.P * math.sqrt(n) / (1 + self.N)

def extract_subgraph_from_mol(mol: Chem.Mol, selected_atoms: Set[int]) -> Tuple[Chem.Mol, List[int]]:
    """Extracts a subgraph from an RDKit molecule given a set of atom indices.

    Parameters
    ----------
    mol : RDKit molecule
        The molecule from which to extract a subgraph.
    selected_atoms : list of int
        The indices of atoms which form the subgraph to be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - RDKit molecule: The subgraph.
        - list of int: Root atom indices from the selected indices.
    """

    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [
            bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]
        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms
            and bond.GetEndAtom().GetIdx() in selected_atoms
        ]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [
        atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms
    ]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    subgraph_mol = new_mol.GetMol()
    
    # Add a sanitization check to ensure chemical validity
    try:
        Chem.SanitizeMol(subgraph_mol)
    except ValueError:
        # If sanitization fails, the molecule is invalid.
        return None, None

    return subgraph_mol, roots


def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[str, List[int]]:
    """Extracts a subgraph from a SMILES given a set of atom indices.

    Parameters
    ----------
    smiles : str
        The SMILES string from which to extract a subgraph.
    selected_atoms : list of int
        The indices of atoms which form the subgraph to be extracted.

    Returns
    -------
    tuple
        A tuple containing:
        - str: SMILES representing the subgraph.
        - list of int: Root atom indices from the selected indices.
    """
    # Attempt 1: with kekulization
    try:
        mol_kekulized = Chem.MolFromSmiles(smiles)
        if mol_kekulized is None: raise ValueError("Invalid input SMILES")
        Chem.Kekulize(mol_kekulized)
        subgraph_kekulized, roots_kekulized = extract_subgraph_from_mol(mol_kekulized, selected_atoms)

        if subgraph_kekulized is not None:
            smiles_kekulized = Chem.MolToSmiles(subgraph_kekulized, kekuleSmiles=True)
            mol_from_smiles = Chem.MolFromSmiles(smiles_kekulized)
            original_mol = Chem.MolFromSmiles(smiles)
            if mol_from_smiles is not None and original_mol is not None and original_mol.HasSubstructMatch(mol_from_smiles):
                return smiles_kekulized, roots_kekulized
    except Exception:
        pass  # This attempt failed, fall through to non-kekulized

    # Attempt 2: without kekulization
    try:
        mol_normal = Chem.MolFromSmiles(smiles)
        if mol_normal is None: raise ValueError("Invalid input SMILES")
        subgraph_normal, roots_normal = extract_subgraph_from_mol(mol_normal, selected_atoms)
        if subgraph_normal is not None:
            smiles_normal = Chem.MolToSmiles(subgraph_normal)
            # Final check to ensure the generated SMILES is valid
            if Chem.MolFromSmiles(smiles_normal) is not None:
                return smiles_normal, roots_normal
    except Exception:
        pass  # This attempt also failed

    return None, None


def mcts_rollout(
    node: MCTSNode,
    state_map: Dict[str, MCTSNode],
    orig_smiles: str,
    clusters: List[Set[int]],
    atom_cls: List[Set[int]],
    nei_cls: List[Set[int]],
    scoring_function: Callable[[List[str]], List[float]],
    min_atoms: int,
    c_puct: float = 10.0,
) -> float:
    """A Monte Carlo Tree Search rollout from a given MCTSNode.

    Parameters
    ----------
    node : MCTSNode
        The MCTSNode from which to begin the rollout.
    state_map : dict
        A mapping from SMILES to MCTSNode.
    orig_smiles : str
        The original SMILES of the molecule.
    clusters : list
        Clusters of atoms.
    atom_cls : list
        Atom indices in the clusters.
    nei_cls : list
        Neighboring cluster indices.
    scoring_function : function
        A function for scoring subgraph SMILES using a Chemprop model.
    min_atoms : int
        The minimum number of atoms in a subgraph.
    c_puct : float
        The constant controlling the level of exploration.

    Returns
    -------
    float
        The score of this MCTS rollout.
    """
    # Return if the number of atoms is less than the minimum
    cur_atoms = node.atoms
    if len(cur_atoms) <= min_atoms:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        # Cluster indices whose all atoms are present in current subgraph
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])

        for i in cur_cls:
            # Leaf atoms are atoms that are only involved in one cluster.
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]

            # This checks
            # 1. If there is only one neighbor cluster in the current subgraph (so that we don't produce unconnected graphs), or
            # 2. If the cluster has only two atoms and the current subgraph has only one leaf atom.
            # If either of the conditions is met, remove the leaf atoms in the current cluster.
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves
        scores = []
        for x in node.children: 
            scores.append(scoring_function([x.smiles]))
            #scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score

    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count, c_puct=c_puct))
    v = mcts_rollout(
        selected_node,
        state_map,
        orig_smiles,
        clusters,
        atom_cls,
        nei_cls,
        scoring_function,
        min_atoms=min_atoms,
        c_puct=c_puct,
    )
    selected_node.W += v
    selected_node.N += 1

    return v



def make_prediction(model,smiles):
    """Makes predictions on a list of SMILES.

    Parameters
    ----------
    model : list
        A model to make predictions with.
    smiles : list
        A SMILES to make predictions on.

    Returns
    -------
    list[list[float]]
       
    """
    #featurizer = featurizers.molecule.V1RDKit2DNormalizedFeaturizer()
    featurizer_ab = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2(),)
    mols = [make_mol(smi, keep_h=False, add_h=False) for smi in smiles]

    datapoints = [MoleculeDatapoint(mol) for mol in mols]
    test_dset = data.MoleculeDataset(datapoints, featurizer = featurizer_ab)
    test_loader = data.build_dataloader(test_dset, shuffle=False, batch_size=1, num_workers=0)
    # Manual prediction loop
    predictions = []
    with torch.no_grad():
        model.eval()
    
        batch_idx = 0 
        for batch in test_loader:
            batch_idx+=1
            pred = model.predict_step(batch, batch_idx)
            predictions.append(pred)
    
    test_preds = torch.cat(predictions, dim=0)
    #sfirst_column = test_preds[:, 0, 0].reshape(-1, 1)  # Access directly
    return test_preds

def mcts(
    smiles: str,
    scoring_function: Callable[[List[str]], List[float]],
    n_rollout: int,
    max_atoms: int,
    prop_delta: float,
    min_atoms: int,
    c_puct: int = 10,
) -> List[MCTSNode]:
    """Runs the Monte Carlo Tree Search algorithm.

    Parameters
    ----------
    smiles : str
        The SMILES of the molecule to perform the search on.
    scoring_function : function
        A function for scoring subgraph SMILES using a Chemprop model.
    n_rollout : int
        The number of MCTS rollouts to perform.
    max_atoms : int
        The maximum number of atoms allowed in an extracted rationale.
    prop_delta : float
        The minimum required property value for a satisfactory rationale.
    min_atoms : int
        The minimum number of atoms in a subgraph.
    c_puct : float
        The constant controlling the level of exploration.

    Returns
    -------
    list
        A list of rationales each represented by a MCTSNode.
    """

    mol = Chem.MolFromSmiles(smiles)

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(
            root,
            state_map,
            smiles,
            clusters,
            atom_cls,
            nei_cls,
            scoring_function,
            min_atoms=min_atoms,
            c_puct=c_puct,
        )

    rationales = [
        node
        for _, node in state_map.items()
        if len(node.atoms) <= max_atoms and node.P >= prop_delta
    ]

    return rationales


def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
 

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2))

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls






def scoring_function(smiles: List[str], model_to_use= None) -> List[float]:
    """
    Modified scoring function to accept the model instance.
    """
    return make_prediction(
        model=model_to_use, # Use the passed model
        smiles=smiles,
    )[0]




###### here's the crucial code that runs the rationale search":
def process_single_smiles(smiles, scoring_func_base, rollout_val, max_atoms_val, prop_delta_val, min_atoms_val, c_puct_val, num_rationales, property_name, model_path_for_worker=None, model=None):
    """Process a single SMILES string to find rationales.
    Loads its own model instance.

    Parameters
    ----------
    smiles : str
        The SMILES string to process.
    scoring_func_base : function
        The base scoring function (without model baked in).
    rollout_val, max_atoms_val, prop_delta_val, min_atoms_val, c_puct_val : various
        Parameters for the MCTS algorithm.
    num_rationales : int
        Number of rationales to keep.
    property_name : str
        Name of the property being interpreted.
    model_path_for_worker : str
        Path to the model file for the worker to load.

    Returns
    -------
    dict
        A dictionary containing the results for this SMILES.
    """
    # <<<--- Load model inside the worker process --->>>
    if torch.cuda.is_available():
        worker_map_location = {'cuda:0': 'cpu'}      
        print(f"Worker process {os.getpid()} loading model to CPU (CUDA available)")
    else:
        worker_map_location = {'cuda:0': 'cpu'}
        print(f"Worker process {os.getpid()} loading model to CPU")

    if model is not None:
        worker_mpnn = model
        worker_mpnn.eval()
        worker_scoring_func = scoring_func_base
    else:
        try:
            worker_mpnn = load_mpnn_model(model_path_for_worker, map_location=worker_map_location)
            worker_mpnn.eval() # Ensure model is in eval mode
        except Exception as e:
            print(f"Error loading model in worker {os.getpid()}: {e}")
            # Handle error appropriately, maybe return an error dict
            return {"smiles": smiles, "error": str(e)}

    # Create the scoring function specific to this worker's model instance
        worker_scoring_func = lambda s: scoring_func_base(s, model_to_use=worker_mpnn)
    # <<<-------------------------------------------->>>

    result = {"smiles": smiles}

    # Get the score for this SMILES using the worker's model
    # Use the worker-specific scoring function
    score = worker_scoring_func([smiles])
    result[property_name] = score

    # Find rationales if score is above threshold
    if score > prop_delta_val:
        rationales = mcts(
            smiles=smiles,
            scoring_function=worker_scoring_func, # Pass the worker's scoring func
            n_rollout=rollout_val,
            max_atoms=max_atoms_val,
            prop_delta=prop_delta_val,
            min_atoms=min_atoms_val,
            c_puct=c_puct_val,
        )
    else:
        rationales = []
    
    # Process rationales
    if len(rationales) == 0:
        for i in range(num_rationales):
            result[f"rationale_{i}"] = None
            result[f"rationale_{i}_score"] = None
    else:
        min_size = min(len(x.atoms) for x in rationales)
        #min_rationales = [x for x in rationales if len(x.atoms) == min_size]
        rats = sorted(rationales, key=lambda x: x.P, reverse=True)
        existing_fps = set() 
        fpgen = AllChem.GetMorganGenerator(radius=2)
        skipped_for_indexing = 0

        for i in range(num_rationales):
            if i < len(rats):
                mol = Chem.MolFromSmiles(rats[i].smiles)
                # Check if molecule is similar to any existing molecule
                fp1 = fpgen.GetFingerprint(mol)
                isSimilar = False
                for existing_fp in existing_fps:
                    if existing_fp is not None and fp1 is not None:
                        similarity = Chem.DataStructs.TanimotoSimilarity(existing_fp, fp1)
                        if similarity >= 0.4:
                            isSimilar = True

                            break
                
                if not isSimilar:
                    existing_fps.add(fp1)
                    result[f"rationale_{i-skipped_for_indexing}"] = rats[i].smiles
                    result[f"rationale_{i-skipped_for_indexing}_score"] = rats[i].P

                else:
                    # Skip this rationale as it's too similar to an existing one
                    skipped_for_indexing += 1
                    continue


            else:
                result[f"rationale_{i}"] = None
                result[f"rationale_{i}_score"] = None
    
    return result

def run_MCTS(smiles_list, output_dir='rationale_finding', num_cores=None, property_name="Y", model_path=None, model=None, disable_multiprocessing=False):
    """
    Run Monte Carlo Tree Search to find rationales for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings to process
    output_dir : str
        Directory to save output files
    num_cores : int
        Number of CPU cores to use (ignored if disable_multiprocessing=True)
    property_name : str
        Name of the property being interpreted
    model_path : str, optional
        Path to the model file to use for predictions. If None and model is None, uses the global model_path
    model : MPNN, optional
        Pre-loaded model object to use for predictions. Takes precedence over model_path if both are provided
    disable_multiprocessing : bool
        If True, uses single-threaded processing regardless of number of molecules
    """
    # Determine which model to use
    if model is not None:
        # Use the provided model object
        use_provided_model = True
        worker_model = model
    elif model_path is not None:
        # Load model from provided path
        use_provided_model = False
        actual_model_path = model_path
    else:
        # Use global model_path
        raise ValueError("Either model or model_path must be provided")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # For very small numbers of molecules or if multiprocessing is disabled, use single processing
    if len(smiles_list) <= 5 or disable_multiprocessing:
        print(f"Using single processing for {len(smiles_list)} molecules.")
        results = []
        
        # Load or use the model for single processing
        if use_provided_model:
            single_core_scoring_func = lambda s: scoring_function(s, model_to_use=worker_model)
        else:
            if torch.cuda.is_available():
                worker_mpnn = load_mpnn_model(actual_model_path, map_location={'cuda:0': 'cuda:0'})
            else:
                worker_mpnn = load_mpnn_model(actual_model_path, map_location={'cuda:0': 'cpu'})
            single_core_scoring_func = lambda s: scoring_function(s, model_to_use=worker_mpnn)
        
        for smiles in tqdm(smiles_list):
            result = process_single_smiles(
                smiles=smiles,
                scoring_func_base=single_core_scoring_func,
                rollout_val=rollout,
                max_atoms_val=max_atoms,
                prop_delta_val=prop_delta,
                min_atoms_val=min_atoms,
                c_puct_val=c_puct,
                num_rationales=num_rationales_to_keep,
                property_name=property_name,
                model=model,
                model_path_for_worker=actual_model_path if not use_provided_model else None
            )
            results.append(result)

        final_df = pd.DataFrame(results)
        # Sort the final DataFrame by property prediction scores in descending order (highest first)
        if property_name in final_df.columns:
            final_df = final_df.sort_values(by=property_name, ascending=False).reset_index(drop=True)
            print(f"Sorted final results by {property_name} scores (highest first)")
        
        # Save final result
        final_df.to_csv(f'{output_dir}/rationale_search_results_final.csv', index=False)
        return final_df

    # For multiprocessing, we need to use model_path (can't pickle model objects easily)
    if use_provided_model:
        raise ValueError("Cannot use multiprocessing with a pre-loaded model object. "
                        "Either provide model_path instead of model, or use disable_multiprocessing=True.")

    # Determine number of cores to use
    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    # For modest numbers of molecules, limit the number of processes to avoid overhead
    if len(smiles_list) < 50:
        num_cores = min(num_cores, max(1, len(smiles_list) // 2))

    print(f"Using {num_cores} CPU cores for parallel processing of {len(smiles_list)} molecules")

    # Create the partial function with fixed arguments
    process_func = partial(
        process_single_smiles,
        scoring_func_base=scoring_function,
        rollout_val=rollout,
        max_atoms_val=max_atoms,
        prop_delta_val=prop_delta,
        min_atoms_val=min_atoms,
        c_puct_val=c_puct,
        num_rationales=num_rationales_to_keep,
        property_name=property_name,
        model = model 
    )

    # Use an appropriate chunk size for efficient parallelism
    chunk_size = max(1, min(len(smiles_list) // (num_cores * 4), 10))
    print(f"Using chunk size: {chunk_size}")

    # Process molecules in parallel
    from threading import Lock
    results_list = []
    lock = Lock()

    ctx = multiprocessing.get_context()
    with ctx.Pool(processes=num_cores) as pool:
        with tqdm(total=len(smiles_list)) as pbar:
            for result in pool.imap_unordered(process_func, smiles_list, chunksize=chunk_size):
                with lock:
                    # Check if the result is an error dictionary
                    if isinstance(result, dict) and "error" in result:
                        print(f"Error processing SMILES {result.get('smiles', 'UNKNOWN')}: {result['error']}")
                    else:
                        results_list.append(result)

                pbar.update(1)

    final_df = pd.DataFrame(results_list)
    
    # Sort the final DataFrame by property prediction scores in descending order (highest first)
    if property_name in final_df.columns:
        final_df = final_df.sort_values(by=property_name, ascending=False).reset_index(drop=True)
        print(f"Sorted final results by {property_name} scores (highest first)")
    
    # Save final result
    final_df.to_csv(f'{output_dir}/rationale_search_results_final.csv', index=False)
    return final_df

def main():
    parser = argparse.ArgumentParser(description='Run MCTS rationale extraction on SMILES strings from a CSV file.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file containing SMILES strings.')
    parser.add_argument('output_csv', type=str, help='Path to save the output CSV file with rationales.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file to use for predictions (default: use global model_path)')
    parser.add_argument('--num_cores', type=int, default=None, help='Number of CPU cores to use (default: all available - 1)')
    parser.add_argument('--output_dir', type=str, default='rationale_finding', help='Directory to save output files')
    parser.add_argument('--property_name', type=str, default='Y', help='Name of the property being interpreted')
    parser.add_argument('--disable_multiprocessing', action='store_true', help='Disable multiprocessing and use single-threaded processing')

    args = parser.parse_args()

    # Read the SMILES data from the input CSV file
    smiles_data = pd.read_csv(args.input_csv)
    smiles_list = smiles_data['SMILES'].tolist()  # Assuming the column is named 'SMILES'

    # Run the MCTS process and save the result
    results = run_MCTS(
        smiles_list, 
        output_dir=args.output_dir,
        num_cores=args.num_cores,
        property_name=args.property_name,
        model_path=args.model_path,
        disable_multiprocessing=args.disable_multiprocessing
    )

    results.to_csv(args.output_csv, index=False)
    
if __name__ == "__main__":
    # Remove the spawn setting, we've already set forkserver above
    main()
 