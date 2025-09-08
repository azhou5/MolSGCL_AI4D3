from chemprop_custom import featurizers
import chemprop_custom.data as data
from chemprop_custom.data import MoleculeDatapoint
from chemprop_custom.utils import make_mol
import torch 
import sys
sys.path.append('../')
from pathlib import Path
from typing import Sequence, Union, Tuple, Optional, Dict, Any
import torch.nn as nn_torch
from chemprop import data, featurizers, models, nn
from sklearn.metrics import roc_curve, auc
from chemprop.models.model import MPNN

import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import data, featurizers, models, nn
from typing import Sequence, Optional, Dict, Any
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import data, featurizers, models, nn

def load_chemprop_model(model_path):
    model = MPNN.load_from_file(model_path, map_location=torch.device('cpu'))
    return model

def get_pred_chemprop(model,smiles):
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



def calculate_auroc_from_preds(preds, y):
    """
    Calculate AUROC from predictions and true values.
    """
    fpr, tpr, thresholds = roc_curve(y, preds)
    return auc(fpr, tpr)





def run_chemprop_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_column: str,
    target_columns: Sequence[str],
    batch_norm: bool = True,
    max_epochs: int = 20,
    num_workers: int = 0,
    checkpoint_dir: str = "checkpoints",
    accelerator: str = "auto",
    devices: int = 1,
    message_passing: str = "bond",         # "bond" or "atom"
    aggregation: str = "mean",             # "mean", "sum", or "norm"
    predictor: str = "regression",         # Chemprop predictor registry key
    metric_keys: Sequence[str] = ("rmse", "mae"),
    init_lr: float = 1e-4,
    max_lr: float = 1e-3,
    final_lr: float = 1e-4,
    **trainer_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train a Chemprop model on pre-split data using the standard MPNN pipeline.

    Returns
    -------
    dict with keys:
        model:       trained Chemprop MPNN
        trainer:     PyTorch Lightning Trainer
        test_metrics: list of final evaluation metrics on test set
        scaler:      target normalization scaler
        dataloaders: dict of train/val/test loaders
    """
    # Featurizer
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())

    def df_to_dp(df: pd.DataFrame):
        return [
            data.MoleculeDatapoint.from_smi(smi, y)
            for smi, y in zip(df[smiles_column], df[target_columns].values)
        ]

    train_data = df_to_dp(train_df)
    val_data = df_to_dp(val_df)
    test_data = df_to_dp(test_df)

    # Datasets and normalization
    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data, featurizer)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data, featurizer)

    # Dataloaders
    loader_kw = dict(num_workers=num_workers, batch_size=32, class_balance=True)
    train_loader = data.build_dataloader(train_dset, **loader_kw)
    val_loader = data.build_dataloader(val_dset, shuffle=False, **loader_kw)
    test_loader = data.build_dataloader(test_dset, shuffle=False, **loader_kw)

    # MPNN components
    mp_cls = nn.AtomMessagePassing if message_passing == "atom" else nn.BondMessagePassing
    mp = mp_cls()

    agg_cls = nn.agg.AggregationRegistry[aggregation]
    agg = agg_cls()

    out_tf = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn_cls = nn.PredictorRegistry[predictor]
    ffn = ffn_cls(output_transform=out_tf)

    metrics = [nn.metrics.MetricRegistry[k]() for k in metric_keys]

    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=batch_norm,
        metrics=metrics,
        warmup_epochs=trainer_kwargs.get("warmup_epochs", 2),
        init_lr=trainer_kwargs.get("init_lr", 1e-4),
        max_lr=trainer_kwargs.get("max_lr", 1e-3),
        final_lr=trainer_kwargs.get("final_lr", 1e-4),
        X_d_transform=nn_torch.Identity(),
    )



    # Checkpointing
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        **trainer_kwargs,
    )

    # Train and evaluate
    trainer.fit(model, train_loader, val_loader)
    test_metrics = trainer.test(model, dataloaders=test_loader)

    return {
        "model": model,
        "trainer": trainer,
        "test_metrics": test_metrics,
        "scaler": scaler,
        "dataloaders": {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
    }
