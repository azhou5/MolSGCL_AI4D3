# MolSGCL
Datasets for molecular property prediction are small relative to the vast chemical space, making generalization from limited experiments a central challenge. We present Mol-SGCL -- Molecular Substructure-Guided Contrastive Learning -- a method that shapes the latent space of molecular property prediction models to align with science-based priors. We hypothesize that engineering inductive biases directly into the representation space encourages models to learn chemical principles rather than overfitting to spurious correlations. Concretely, Mol-SGCL employs a triplet loss that pulls a molecule’s representation toward representations of plausibly causal substructures and pushes it away from implausibly causal ones. Plausibility is defined by querying a large language model with the list of extracted substructures. To stress-test out-of-distribution (OOD) generalization under data scarcity, we construct modified Therapeutics Data Commons tasks that minimize train–test similarity and cap the training set at 150 molecules. On these OOD splits, Mol-SGCL outperforms baselines, indicating that Mol-SGCL promotes invariant feature learning and enhances model generalizability in data-limited regimes. We further demonstrate that Mol-SGCL transfers successfully to a D-MPNN backbone, highlighting that the approach is not tied to a specific architecture. We imagine that Mol-SGCL may enhance AI drug discovery by improving molecular property prediction for novel candidate molecules that are out-of-distribution from existing data. Beyond molecular property prediction, we envision that this could be extended to diverse therapeutics tasks, as long as the inputs can be decomposed into substructures whose presence, absence, or configuration has an influence on the target label.



![Overview](./overview.png)


# Repo Structure

*chemprop_environment.yml* and *minimol_environment.yml* can be used to recreate the conda environments used in this study.
The custom data-splits (150 train set size) are in *data*
*chemprop_custom* is a clone of the chemprop package, with substantial changes in the models/model.py (where the triplet loss is defined), as well as the *data/datasets.py, and data/datapoints.py file. 
*code* contains the python implementations of Mol-SGCL, as well as the shell scripts that can be used to run the experiments. 


In *code*:

 
- *minimol_triplet_model.py* is where Minimol-Mol-SGCL model is defined 
- *minimol_triplet_runner.py* is the python file that sets up the evaluation of the Minimol-Mol-SGCL file
- *experiment_minimol_<>.sh* are shell scripts that can be used to run each of the experiments.  
- *model_utils.py* contains several assorted utility functions
- *get_rationale_for_plausibility.py* and *get_rationale_regression.py* run a MCTS to return substructures. Used for the D-MPNN implementation
- *LLM_assisted_plausibility* is a folder that contains:
    - *get_plausibility.py*: This uses GPT5 to produce plausibility labels. 
    - *molecular_description_image.py*: The uses GPT5-mini to generate a natural language description of molecules. 
    - *describe_cli.py*: This is a file that a shell script can use to run the molecular description file due to environment conflicts from the minimol environment. 
- *dmpnn_mol_sgcl.py* is the file to run the DMPNN version of Mol-SGCL 
- *run dmpnn_mol_sgcl_llm.sh* is a bash script that can be used to run dmpnn_mol_sgcl.py for Mol-SGCL_Rules. It is pre-written to run the lipophilicity evaluation.
- *gen_harder_scripts.py* takes in an input Therapeutic Data Commons .py file and generates more challenging train test splitts (capped at 150 molecules). 



# To run Mol-SGCL_Minimol:

1. Create a file in your Add your OpenAI API key to to a .env file in the project root. 
2. Create a dictionary (.pt) mapping the SMILES in your dataset to the minimol fingerprints.
3 Run the desired experiment_minimol_<>.sh file. Make sure to activate your minimol environment. Insert the smiles to fingerprint path in the `cache_path`. 


# To run Mol-SGCL-DMPNN: 

1. Create a file in your Add your OpenAI API key to to a .env file in the project root. 

2. Run the run_dmpnn_mol_sgcl_llm.sh file. Make sure to activate your chemprop environment. 






