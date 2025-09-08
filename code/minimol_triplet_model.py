import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import os
from lightning import pytorch as pl

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CachedEncoder:
    def __init__(self, encoder, cache_file: str):
        self.encoder = encoder

        self.cache_file = cache_file
        self.cache = {}
        if os.path.isfile(cache_file):
            try:
                self.cache = torch.load(cache_file)
                if not isinstance(self.cache, dict):
                    raise ValueError("cache object is not a dict")
                print(f"✓ Loaded cache ({len(self.cache)} SMILES)")
            except Exception as e:
                print(f"Warning: failed to load cache from {cache_file}: {e}. Starting fresh cache.")
                self.cache = {}
        else:
            print("✗ No cache found, starting empty")

    def encode(self, smiles_list, chunk_size=256):
        missing = [s for s in smiles_list if s not in self.cache]
        if missing:
            for i in range(0, len(missing), chunk_size):
                chunk = missing[i:i+chunk_size]
                try:
                    embeddings = self.encoder(chunk)
                    for smi, feat in zip(chunk, embeddings):
                        self.cache[smi] = feat.cpu()
                except:
                    for smi in chunk:
                        try:
                            self.cache[smi] = self.encoder([smi])[0].cpu()
                        except:
                            self.cache[smi] = torch.zeros(512)
            # Ensure directory exists
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            try:
                torch.save(self.cache, self.cache_file)
            except Exception as e:
                print(f"Warning: failed to save cache to {self.cache_file}: {e}")
        return torch.stack([self.cache[s] for s in smiles_list], dim=0).to(DEVICE)

class MinimolTripletNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        repr_dim: int = 512,
        hidden_dim: int = 512,
        depth: int = 3,
        combine: bool = False,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        if depth not in (3, 4):
            raise ValueError("depth must be 3 or 4")
        self.input_dim = int(input_dim)
        self.repr_dim = int(repr_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.combine = bool(combine)
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout_p = float(dropout)

        # Representation layer: always first after the fingerprint
        self.repr = nn.Linear(self.input_dim, self.repr_dim)
        self.bn_repr = nn.BatchNorm1d(self.repr_dim)

        # Hidden layers after representation
        self.dense1 = nn.Linear(self.repr_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)

       # self.dense2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        if self.depth == 4:
            self.dense2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        # Optionally disable BatchNorm layers
        if not self.use_batch_norm:
            self.bn_repr = nn.Identity()
            self.bn1 = nn.Identity()
            if self.depth == 4:
                self.bn2 = nn.Identity()

        final_in = self.hidden_dim + (self.input_dim if self.combine else 0)
        self.final_dense = nn.Linear(final_in, output_dim)

    def encode(self, x: torch.Tensor, apply_dropout_relu_bn: bool = True) -> torch.Tensor:
        # Return representation embedding (first layer after fingerprint)
        x = self.repr(x)

        if apply_dropout_relu_bn:
            x = self.bn_repr(x)
            x = F.relu(x)
            x = self.dropout(x)
        else: 
            x = F.relu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        # Representation
        x = self.encode(x, apply_dropout_relu_bn=True)
        # we do this because when we calculate the triplet loss, we want to use the raw representation, without dropout. 
        # Remaining hidden layers
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.depth == 4:
            x = self.dense2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = torch.cat((x, original_x), dim=1) if self.combine else x
        x = self.final_dense(x)
        return x.squeeze(-1)


class CosineTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, pos_dis_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.pos_dis_weight = pos_dis_weight

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)
        dist_pos = 1.0 - sim_pos
        dist_neg = 1.0 - sim_neg
        loss = torch.relu(self.pos_dis_weight * dist_pos - dist_neg + self.margin)
        return loss.mean()


class MinimolTripletModule(pl.LightningModule):
    def __init__(
        self,
        is_regression: bool = False,
        margin: float = 0.2,
        triplet_weight: float = 1.0,
        init_lr: float = 1e-3,
        input_dim: int = 512,
        repr_dim: int = 512,
        hidden_dim: int = 512,
        depth: int = 3,
        combine: bool = False,
        use_batch_norm: bool = True,
        dropout: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = MinimolTripletNet(
            input_dim=input_dim,
            repr_dim=repr_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            combine=combine,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            output_dim=1,
        )
        self.is_regression = is_regression
        self.margin = margin
        self.triplet_weight = triplet_weight
        self.init_lr = init_lr
        self.triplet_criterion = CosineTripletLoss(margin=margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _main_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            return F.mse_loss(pred, target.float(), reduction='mean')
        return F.binary_cross_entropy_with_logits(pred, target.float(), reduction='mean')

    def _rep_loss(self, anchor_x: torch.Tensor, pos_x: torch.Tensor, neg_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum().item() <= 1:
            return torch.tensor(0.0, device=anchor_x.device)
        # Use representation without dropout for triplet loss
        a = self.net.encode(anchor_x[mask], apply_dropout_relu_bn=False)
        p = self.net.encode(pos_x[mask], apply_dropout_relu_bn=False)
        n = self.net.encode(neg_x[mask], apply_dropout_relu_bn=False)
        return self.triplet_criterion(a, p, n)

    def training_step(self, batch, batch_idx: int):
        x, y = batch['x'], batch['y']
        pred = self(x)
        main_loss = self._main_loss(pred.squeeze(), y)
        rep_loss = self._rep_loss(batch['x'], batch['pos_x'], batch['neg_x'], batch['has_triplet'])
        loss = main_loss + self.triplet_weight * rep_loss
        self.log('train_main_loss', main_loss, prog_bar=True, on_epoch=True)
        self.log('train_triplet_loss', rep_loss, prog_bar=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch['x'], batch['y']
        pred = self(x)
        main_loss = self._main_loss(pred.squeeze(), y)
        rep_loss = self._rep_loss(batch['x'], batch['pos_x'], batch['neg_x'], batch['has_triplet'])
        loss = main_loss + self.triplet_weight * rep_loss 
        self.log('val_main_loss', main_loss, prog_bar=True, on_epoch=True)
        self.log('val_triplet_loss', rep_loss, prog_bar=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)


def select_plausible_implausible(rationales: List[str], plausibility: List[int]) -> Tuple[Optional[str], Optional[str]]:
    pl = np.array(plausibility)
    if (pl == 1).any() and (pl == -1).any():
        pos_idx = int(np.where(pl == 1)[0][0])
        neg_idx = int(np.where(pl == -1)[0][0])
        return rationales[pos_idx], rationales[neg_idx]
    if (pl == -1).sum() == 1 and ((pl == 0) | (pl == -1)).all():
        neg_idx = int(np.where(pl == -1)[0][0])
        zero_idxs = np.where(pl == 0)[0]
        if len(zero_idxs) > 0:
            pos_idx = int(np.random.choice(list(zero_idxs)))
            return rationales[pos_idx], rationales[neg_idx]
    return None, None


class MinimolTripletDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_to_plausibility: Optional[Dict[str, Dict[str, List]]] = None,
        require_triplet: bool = False,
    ):
        super().__init__()
        self.rows = []
        smiles_to_plausibility = smiles_to_plausibility or {}
        for _, row in df.iterrows():
            s = str(row['SMILES'])
            y = float(row['Y'])
            pos, neg = None, None
            pl_entry = smiles_to_plausibility.get(s)
            if pl_entry and 'Rationales' in pl_entry and 'plausibility' in pl_entry:
                pos, neg = select_plausible_implausible(pl_entry['Rationales'], pl_entry['plausibility'])
            if require_triplet and (pos is None or neg is None):
                continue
            self.rows.append((s, y, pos, neg))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def minimol_triplet_collate(encoder: CachedEncoder):
    def _collate(batch: List[Tuple[str, float, Optional[str], Optional[str]]]):
        anchors: List[str] = [b[0] for b in batch]
        ys: List[float] = [b[1] for b in batch]
        pos_list: List[Optional[str]] = [b[2] for b in batch]
        neg_list: List[Optional[str]] = [b[3] for b in batch]

        x = encoder.encode(anchors)

        has_triplet = torch.tensor([p is not None and n is not None for p, n in zip(pos_list, neg_list)], dtype=torch.bool, device=DEVICE)

        pos_to_idx: Dict[int, str] = {i: s for i, s in enumerate(pos_list) if s is not None}
        neg_to_idx: Dict[int, str] = {i: s for i, s in enumerate(neg_list) if s is not None}

        pos_feats = torch.zeros((len(batch), x.size(1)), device=DEVICE)
        neg_feats = torch.zeros((len(batch), x.size(1)), device=DEVICE)

        if len(pos_to_idx) > 0:
            pos_order = [pos_to_idx[i] for i in pos_to_idx]
            pos_encoded = encoder.encode(pos_order)
            for j, i in enumerate(pos_to_idx.keys()):
                pos_feats[i] = pos_encoded[j]

        if len(neg_to_idx) > 0:
            neg_order = [neg_to_idx[i] for i in neg_to_idx]
            neg_encoded = encoder.encode(neg_order)
            for j, i in enumerate(neg_to_idx.keys()):
                neg_feats[i] = neg_encoded[j]

        y = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
        return {
            'x': x,
            'y': y,
            'pos_x': pos_feats,
            'neg_x': neg_feats,
            'has_triplet': has_triplet,
        }

    return _collate


def build_minimol_triplet_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    encoder: CachedEncoder,
    smiles_to_plausibility: Optional[Dict[str, Dict[str, List]]] = None,
    batch_size: int = 128,
    require_triplet_for_train: bool = False,
):
    train_ds = MinimolTripletDataset(train_df, smiles_to_plausibility, require_triplet=require_triplet_for_train)
    val_ds = MinimolTripletDataset(val_df, smiles_to_plausibility=None, require_triplet=False)

    collate = minimol_triplet_collate(encoder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def train_minimol_triplet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    encoder: CachedEncoder,
    is_regression: bool = False,
    margin: float = 0.2,
    triplet_weight: float = 1.0,
    init_lr: float = 1e-3,
    max_epochs: int = 20,
    smiles_to_plausibility: Optional[Dict[str, Dict[str, List]]] = None,
    batch_size: int = 128,
    require_triplet_for_train: bool = False,
    accelerator: Optional[str] = None,
    devices: Optional[int] = None,
    repr_dim: int = 512,
    hidden_dim: int = 512,
    depth: int = 3,
    combine: bool = False,
    use_batch_norm: bool = True,
    dropout: float = 0.1,
):
    model = MinimolTripletModule(
        is_regression=is_regression,
        margin=margin,
        triplet_weight=triplet_weight,
        init_lr=init_lr,
        repr_dim=repr_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        combine=combine,
        use_batch_norm=use_batch_norm,
        dropout=dropout
    )

    train_loader, val_loader = build_minimol_triplet_dataloaders(
        train_df=train_df,
        val_df=val_df,
        encoder=encoder,
        smiles_to_plausibility=smiles_to_plausibility,
        batch_size=batch_size,
        require_triplet_for_train=require_triplet_for_train,
    )

    chosen_accelerator = accelerator if accelerator else ('gpu' if torch.cuda.is_available() else 'cpu')
    chosen_devices = devices if devices is not None else 1
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=chosen_accelerator,
        devices=chosen_devices,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)
    return model


