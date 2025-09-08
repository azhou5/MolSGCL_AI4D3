from __future__ import annotations

import io
from typing import Iterable
import warnings

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim
import numpy as np

from chemprop_custom.data import BatchMolGraph, TrainingBatch
from chemprop_custom.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop_custom.nn.transforms import ScaleTransform
from chemprop_custom.schedulers import build_NoamLike_LRSched
from chemprop_custom.data import BatchMolGraph, TrainingBatch
from chemprop_custom.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop_custom.nn.transforms import ScaleTransform
from chemprop_custom.schedulers import build_NoamLike_LRSched
import torch.nn.functional as F
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def cosine_similarity(self, vec1, vec2):
        # Compute cosine similarity using PyTorch operations
        dot_product = torch.sum(vec1 * vec2, dim=1)
        norm_vec1 = torch.norm(vec1, dim=1)
        norm_vec2 = torch.norm(vec2, dim=1)
        
        # Handle zero division
        zeros = (norm_vec1 == 0) | (norm_vec2 == 0)
        similarity = torch.zeros_like(dot_product)
        valid = ~zeros
        similarity[valid] = dot_product[valid] / (norm_vec1[valid] * norm_vec2[valid])
        
        return similarity

    def euclidean_distance(self, v1, v2):
        """Calculate Euclidean (L2) distance between two vectors using PyTorch."""
        return torch.sqrt(torch.sum((v1 - v2) ** 2, dim=1))

    def forward(self, output, rationales, plausible):
        #similarities = self.cosine_similarity(output, rationales)
        similarities = 1 / (1 + self.euclidean_distance(output, rationales))
        # Calculate class weights based on number of samples in each class
        n_plausible = plausible.sum()
        n_implausible = (~plausible).sum()
        total = n_plausible + n_implausible
        print(n_implausible)
        # Compute balanced weights (inverse frequency)
        plausible_weight = total / (2 * n_plausible) if n_plausible > 0 else 0
        implausible_weight = total / (2 * n_implausible) if n_implausible > 0 else 0
        
        # Apply weighted loss
        loss = torch.where(
            plausible,
            plausible_weight * (1 - similarities),    # Weighted loss for plausible
            implausible_weight * similarities         # Weighted loss for implausible
        )
        print("loss", loss)
        return torch.sum(loss)
    





class TripletEncoderTrainer(pl.LightningModule):
    def __init__(self, mpnn: MPNN, warmup_epochs: int = 2, lr: float = 2e-4, margin: float = 1.0):
        super().__init__()
        self.mpnn = mpnn
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.init_lr = lr
        self.criterion = TripletLoss(margin=margin)
        self.max_lr: float =10*lr
        self.final_lr: float =lr
        
    def forward(self, batch: TrainingBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Generate encodings for molecule and both rationales."""
        main_encoding = self.mpnn.fingerprint(batch.bmg, batch.V_d, batch.X_d)
        pos_rationale_encoding = self.mpnn.fingerprint(batch.rationale_bmg, batch.V_d, batch.X_d)
        neg_rationale_encoding = self.mpnn.fingerprint(batch.neg_rationale_bmg, batch.V_d, batch.X_d)
        return main_encoding, pos_rationale_encoding, neg_rationale_encoding

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Compute training loss for a batch."""
        main_encoding, pos_encoding, neg_encoding = self(batch)
        loss = self.criterion(main_encoding, pos_encoding, neg_encoding)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Compute validation loss for a batch."""
        main_encoding, pos_encoding, neg_encoding = self(batch)
        loss = self.criterion(main_encoding, pos_encoding, neg_encoding)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * self.warmup_epochs * self.trainer.num_training_batches
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * self.trainer.num_training_batches
            
        scheduler = build_NoamLike_LRSched(
            optimizer,
            warmup_steps=self.warmup_epochs * self.trainer.num_training_batches,
            cooldown_steps=cooldown_steps,
            init_lr=self.init_lr,
            max_lr=self.max_lr,
            final_lr=self.final_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

class EncoderTrainer(pl.LightningModule):
    """Trainer class for encoding molecules and their rationales.
    
    This trainer learns to encode molecules such that plausible rationales have similar 
    encodings to their parent molecules, while implausible rationales have different encodings.
    
    Parameters
    ----------
    mpnn : MPNN
        The message passing neural network model to use for encoding molecules
    warmup_epochs : int, default=2
        Number of warmup epochs for the learning rate scheduler
    init_lr : float, default=1e-4
        Initial learning rate
    """
    def __init__(self, mpnn: MPNN, warmup_epochs: int = 2, init_lr: float = 1e-4):
        super().__init__()
        self.mpnn = mpnn
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.criterion = CustomLoss()
        self.max_lr: float = 5e-3
        self.final_lr: float = 1e-4
    def forward(self, batch: TrainingBatch) -> tuple[Tensor, Tensor]:
        """Generate encodings for both main molecule and rationale."""
        main_encoding = self.mpnn.encoding(batch.bmg, batch.V_d, batch.X_d)
        rationale_encoding = self.mpnn.encoding(batch.rationale_bmg, batch.V_d, batch.X_d)
        return main_encoding, rationale_encoding
        
    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Compute training loss for a batch."""
        main_encoding, rationale_encoding = self(batch)
        loss = self.criterion(main_encoding, rationale_encoding, batch.plausible)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Compute validation loss for a batch."""
        main_encoding, rationale_encoding = self(batch)
        loss = self.criterion(main_encoding, rationale_encoding, batch.plausible)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * self.warmup_epochs * self.trainer.num_training_batches
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * self.trainer.num_training_batches
            
        scheduler = build_NoamLike_LRSched(
            optimizer,
            warmup_steps=self.warmup_epochs * self.trainer.num_training_batches,
            cooldown_steps=cooldown_steps,
            init_lr=self.init_lr,
            max_lr=self.max_lr,
            final_lr=self.final_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

class MPNN(pl.LightningModule):
    r"""An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine.

    The first two modules calculate learned fingerprints from an input molecule
    reaction graph, and the final module takes these learned fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{MPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MessagePassing
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level predictor
    predictor : Predictor
        the function to use to calculate the final prediction
    batch_norm : bool, default=False
        if `True`, apply batch normalization to the output of the aggregation operation
    metrics : Iterable[Metric] | None, default=None
        the metrics to use to evaluate the model during training and evaluation
    warmup_epochs : int, default=2
        the number of epochs to use for the learning rate warmup
    init_lr : int, default=1e-4
        the initial learning rate
    max_lr : float, default=1e-3
        the maximum learning rate
    final_lr : float, default=1e-4
        the final learning rate

    Raises
    ------
    ValueError
        if the output dimension of the message passing block does not match the input dimension of
        the predictor function
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        epochs: int = None, 
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        adversarial_weight: float = 0.3,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        # manually add X_d_transform to hparams to suppress lightning's warning about double saving
        # its state_dict values.
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "agg", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = 0
    ) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation"""
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        Z = self.fingerprint(bmg, V_d, X_d)
        return self.predictor(Z)

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, prog_bar=True, on_epoch=True, batch_size=batch_size)

        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()
    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: TrainingBatch, label: str) -> None:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=len(batch[0]))

    def predict_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Generate predictions for the input molecules/reactions during inference"""
        bmg, V_d, X_d = batch.bmg, batch.V_d, batch.X_d
        Z = self.fingerprint(bmg, V_d, X_d)
        return self.predictor(Z)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if not hasattr(submodules["predictor"].criterion, "_defaults"):
            submodules["predictor"].criterion = submodules["predictor"].criterion.__class__(
                task_weights=submodules["predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location)
        d["state_dict"] = state_dict
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model

def train_encoder(mpnn: MPNN, train_loader, val_loader, max_epochs=10):
    encoder_trainer = EncoderTrainer(mpnn)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(encoder_trainer, train_loader, val_loader)
    return mpnn

def train_triplet_encoder(mpnn: MPNN, train_loader, val_loader, max_epochs=10, margin=1.0, lr=2e-4):
    encoder_trainer = TripletEncoderTrainer(mpnn, margin=margin, lr=lr)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(encoder_trainer, train_loader, val_loader)
    return mpnn
def train_triplet_encoder_bce(mpnn: MPNN, train_loader, val_loader, max_epochs=10, margin=1.0, triplet_weight=1.0, init_lr=2e-4, max_lr=2e-3, final_lr=2e-4, is_regression=False):
    encoder_trainer = TripletEncoderTrainerWithBCE(mpnn, margin=margin, triplet_weight=triplet_weight, init_lr=init_lr, max_lr=max_lr, final_lr=final_lr, is_regression=is_regression)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(encoder_trainer, train_loader, val_loader)
    trained_mpnn = encoder_trainer.mpnn
    del trainer
    return trained_mpnn
def train_adversarial_mpnn(mpnn: MPNN, train_loader, val_loader, max_epochs=10, init_lr=2e-4, max_lr=2e-3, final_lr=2e-4, adversarial_weight=0.3):
    adversarial_trainer = AdversarialMPNNTrainer(mpnn, init_lr=init_lr, max_lr=max_lr, final_lr=final_lr, adversarial_weight=adversarial_weight)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(adversarial_trainer, train_loader, val_loader)
    return mpnn


def create_datapoint_with_rationale(smiles: str, rationale_smiles: str, plausible: bool) -> MoleculeDatapoint:
    return MoleculeDatapoint(
        mol=Chem.MolFromSmiles(smiles),
        rationale_mol=Chem.MolFromSmiles(rationale_smiles),
        plausible=plausible
    )



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss between anchor (molecule), positive (good rationale),
        and negative (bad rationale) encodings
        """
        # Calculate distances
        dist_pos = torch.norm(anchor - positive, dim=1)
        dist_neg = torch.norm(anchor - negative, dim=1)
        
        # Compute triplet loss with margin
        #loss = dist_pos - torch.pow(torch.relu(self.margin - dist_neg), 2)
        #loss = torch.pow(dist_pos, 2) - torch.pow(dist_neg, 2)


        loss  =  torch.relu(dist_pos - dist_neg+ 7)

        return torch.mean(loss)
class CosineTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.pos_dis_weight = 1.0
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss between anchor (molecule), positive (good rationale),
        and negative (bad rationale) encodings using cosine similarity.
        """
        # Calculating cosine similarities
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)
        dist_pos = 1.0 - sim_pos
        dist_neg = 1.0 - sim_neg
        # Compute triplet loss with margin and positive distance weighting - FIXED VERSION
        # We want dist_pos < dist_neg, so we penalize when pos_dis_weight*dist_pos - dist_neg + margin > 0
        loss = torch.relu(self.pos_dis_weight * dist_pos - dist_neg + self.margin)
        
        return torch.mean(loss)


class TripletEncoderTrainerWithBCE(pl.LightningModule):
    def __init__(self, mpnn: MPNN, warmup_epochs: int = 2, init_lr: float = 2e-4,
                 margin: float = 1.0, triplet_weight: float = 1.0, max_lr: float = 2e-3,
                 final_lr: float = 2e-4, is_regression: bool = False):
        super().__init__()
        self.mpnn = mpnn
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        # store hyperparameters
        self.triplet_weight = triplet_weight
        self.margin = margin
        self.is_regression = is_regression
        # pure representation triplet loss
        self.rep_criterion = CosineTripletLoss(margin=margin)
        self.max_lr = max_lr
        self.final_lr = final_lr

    def forward(self, batch: TrainingBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Return main encoding, prediction logit, and target for the batch."""
        main_encoding = self.mpnn.fingerprint(batch.bmg, batch.V_d, batch.X_d)
        #prediction = self.mpnn.predictor(main_encoding)[:,0, 0].reshape(-1, 1)

        prediction = self.mpnn.predictor(main_encoding)[:, 0].reshape(-1, 1)
        target = batch.Y.squeeze()
        return main_encoding, prediction, target

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        main_enc, prediction, target = self(batch)

        if self.is_regression:
            main_loss = F.mse_loss(
                prediction.squeeze(), target.float(), reduction='mean'
            )
        else:
            # 1) weighted BCE over all samples
            total_samples = len(target)
            pos_count = (target == 1.0).sum() 
            neg_count = (target == 0.0).sum()

            if pos_count.item() > 0:
                weight_for_positive_class = total_samples / (2.0 * pos_count.float()) 
            else:
                weight_for_positive_class = torch.tensor(0.0, device=target.device, dtype=target.dtype)

            if neg_count.item() > 0:
                weight_for_negative_class = total_samples / (2.0 * neg_count.float())
            else:
                weight_for_negative_class = torch.tensor(0.0, device=target.device, dtype=target.dtype)
            
            weights = torch.where(target == 1.0, weight_for_positive_class, weight_for_negative_class)
            
            main_loss = F.binary_cross_entropy_with_logits(
                prediction.squeeze(), target.float(), weight=weights, reduction='mean'
            )

        print(f"len of the target: {len(batch.Y)}")
        
        # 2) pure triplet representation loss for samples with rationales
        rep_loss = torch.tensor(0.0, device=main_enc.device) # Default to zero
        # Use the mask passed in the batch
        if batch.has_rationale_mask is not None and batch.rationale_bmg is not None:
             mask = batch.has_rationale_mask
             print(f"sum of the masks: {mask.sum()}")
             print(f"len of the mask: {len(mask)}")
             
             # Log rationale balance verification
             num_with_rationales = mask.sum().item()
             num_without_rationales = len(mask) - num_with_rationales
             batch_size = len(mask)
             rationale_fraction = num_with_rationales / batch_size if batch_size > 0 else 0.0
             print(f"Batch rationale balance: {num_with_rationales}/{batch_size} = {rationale_fraction:.3f} with rationales")
             
             # Also check class balance within rationale/non-rationale groups
             if num_with_rationales > 0:
                 rationale_targets = target[mask]
                 rationale_pos = (rationale_targets == 1.0).sum().item()
                 rationale_neg = (rationale_targets == 0.0).sum().item()
                 print(f"Among rationale samples: {rationale_pos} pos, {rationale_neg} neg")
             
             if num_without_rationales > 0:
                 non_rationale_targets = target[~mask]
                 non_rationale_pos = (non_rationale_targets == 1.0).sum().item()
                 non_rationale_neg = (non_rationale_targets == 0.0).sum().item()
                 print(f"Among non-rationale samples: {non_rationale_pos} pos, {non_rationale_neg} neg")
             if mask.sum() > 1:
                # Ensure neg_rationale_bmg also exists if mask is True
                if batch.neg_rationale_bmg is None:
                     raise ValueError("Inconsistent state: has_rationale_mask is True but neg_rationale_bmg is None")

                # Filter X_d for the rationale samples if X_d exists
                X_d_for_rationales = batch.X_d[mask] if batch.X_d is not None else None
                # batch.V_d is expected to be None based on MoleculeDatapoint creation

                pos_small = self.mpnn.fingerprint(batch.rationale_bmg, batch.V_d, X_d_for_rationales)
                neg_small = self.mpnn.fingerprint(batch.neg_rationale_bmg, batch.V_d, X_d_for_rationales)
                # Apply triplet loss only to the samples indicated by the mask
                rep_loss = self.rep_criterion(main_enc[mask], pos_small, neg_small)

        loss = main_loss + rep_loss * self.triplet_weight
        print(f'the main loss is {main_loss.item()}. The triplet loss is {rep_loss.item()}. The total loss is {loss.item()}')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        main_enc, prediction, target = self(batch)

        if self.is_regression:
            main_loss = F.mse_loss(
                prediction.squeeze(), target.float(), reduction='mean'
            )
        else:
            # 1) weighted BCE over all samples
            total_samples = len(target)
            pos_count = (target == 1.0).sum()
            neg_count = (target == 0.0).sum()

            if pos_count.item() > 0:
                weight_for_positive_class = total_samples / (2.0 * pos_count.float())
            else:
                weight_for_positive_class = torch.tensor(0.0, device=target.device, dtype=target.dtype)

            if neg_count.item() > 0:
                weight_for_negative_class = total_samples / (2.0 * neg_count.float())
            else:
                weight_for_negative_class = torch.tensor(0.0, device=target.device, dtype=target.dtype)

            weights = torch.where(target == 1.0, weight_for_positive_class, weight_for_negative_class)
            
            main_loss = F.binary_cross_entropy_with_logits(
                prediction.squeeze(), target.float(), weight=weights, reduction='mean'
            )

        # 2) pure triplet representation loss for samples with rationales
        rep_loss = torch.tensor(0.0, device=main_enc.device) # Default to zero
        # Use the mask passed in the batch
        if batch.has_rationale_mask is not None and batch.rationale_bmg is not None:
             mask = batch.has_rationale_mask
             
             # Log rationale balance verification (validation)
             num_with_rationales = mask.sum().item()
             batch_size = len(mask)
             rationale_fraction = num_with_rationales / batch_size if batch_size > 0 else 0.0
             print(f"Val batch rationale balance: {num_with_rationales}/{batch_size} = {rationale_fraction:.3f} with rationales")
             
             if mask.sum() > 1:
                # Ensure neg_rationale_bmg also exists if mask is True
                if batch.neg_rationale_bmg is None:
                     raise ValueError("Inconsistent state: has_rationale_mask is True but neg_rationale_bmg is None")

                # Filter X_d for the rationale samples if X_d exists
                X_d_for_rationales = batch.X_d[mask] if batch.X_d is not None else None
                # batch.V_d is expected to be None based on MoleculeDatapoint creation

                pos_small = self.mpnn.fingerprint(batch.rationale_bmg, batch.V_d, X_d_for_rationales)
                neg_small = self.mpnn.fingerprint(batch.neg_rationale_bmg, batch.V_d, X_d_for_rationales)
                # Apply triplet loss only to the samples indicated by the mask
                rep_loss = self.rep_criterion(main_enc[mask], pos_small, neg_small)

        loss = main_loss + rep_loss * self.triplet_weight

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * self.warmup_epochs * self.trainer.num_training_batches
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * self.trainer.num_training_batches
            
        scheduler = build_NoamLike_LRSched(
            optimizer,
            warmup_steps=self.warmup_epochs * self.trainer.num_training_batches,
            cooldown_steps=cooldown_steps,
            init_lr=self.init_lr,
            max_lr=self.max_lr,
            final_lr=self.final_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }




class AdversarialMPNNTrainer(pl.LightningModule):
    """Trainer class for adversarial training of MPNN models.
    
    This trainer learns to encode molecules such that the main task performance is maintained
    while being adversarial to a secondary prediction task.
    
    Parameters
    ----------
    mpnn : MPNN
        The pretrained message passing neural network model
    warmup_epochs : int, default=2
        Number of warmup epochs for the learning rate scheduler
    init_lr : float, default=1e-4
        Initial learning rate
    adversarial_weight : float, default=0.3
        Weight for the adversarial component of the loss
    """
    def __init__(
        self, 
        mpnn: MPNN, 
        warmup_epochs: int = 2, 
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        adversarial_weight: float = 0.3
    ):
        super().__init__()
        self.mpnn = mpnn
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.adversarial_weight = adversarial_weight
        
        # Get input size from the predictor's first layer
        fingerprint_size = mpnn.predictor.input_dim
        
        # Create secondary predictor with same architecture but different weights
        self.secondary_predictor = nn.Sequential(
            nn.Linear(fingerprint_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1)  # For secondary target prediction
        )
        
    def forward(self, batch: TrainingBatch) -> tuple[Tensor, Tensor]:
        """Generate predictions for both main task and secondary task."""
        # Get molecule fingerprints
        Z = self.mpnn.fingerprint(batch.bmg, batch.V_d, batch.X_d)
        
        # Generate predictions
        main_preds = self.mpnn.predictor(Z)
        secondary_preds = self.secondary_predictor(Z)
        
        return main_preds, secondary_preds
        
    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Compute training loss for a batch."""
        main_preds, secondary_preds = self(batch)
        
        # Main task loss
        # Ensure main_preds has correct shape by selecting first output if needed
        main_preds = main_preds.squeeze()
        if len(main_preds.shape) > 1:
            main_preds = main_preds[:, 0]  # Take first output if multiple outputs exist
        
        total_samples = len(batch.Y)
        pos_samples = (batch.Y == 1).sum()
        neg_samples = (batch.Y == 0).sum()
        pos_weight = total_samples / (2 * pos_samples) if pos_samples > 0 else 0
        neg_weight = total_samples / (2 * neg_samples) if neg_samples > 0 else 0
        
        # Create weight tensor for each sample
        weights = torch.where(batch.Y == 1, 
                            torch.tensor(pos_weight, device=batch.Y.device),
                            torch.tensor(neg_weight, device=batch.Y.device))
        
        # Calculate weighted BCE loss with properly shaped tensors
        bce_loss = F.binary_cross_entropy_with_logits(
            main_preds, 
            batch.Y.float().squeeze(),
            weight=weights,
            reduction='mean'
        )
        
        # Secondary task loss (MSE)
        secondary_loss = F.mse_loss(
            secondary_preds.squeeze(),
            batch.secondary_target.float().squeeze(),
            reduction='mean'
        )
        
        # Total loss with gradient reversal for secondary prediction
        total_loss = bce_loss - torch.min(self.adversarial_weight * secondary_loss, torch.tensor(150.0, device=bce_loss.device))
        
        # Log metrics
        self.log('train_main_loss', bce_loss, prog_bar=True)
        self.log('train_secondary_loss', secondary_loss, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)
        
        return total_loss
        
    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """Compute validation loss for a batch."""
        main_preds, secondary_preds = self(batch)
        
        # Main task loss
        # Ensure main_preds has correct shape by selecting first output if needed
        main_preds = main_preds.squeeze()
        if len(main_preds.shape) > 1:
            main_preds = main_preds[:, 0]  # Take first output if multiple outputs exist
        
        total_samples = len(batch.Y)
        pos_samples = (batch.Y == 1).sum()
        neg_samples = (batch.Y == 0).sum()
        pos_weight = total_samples / (2 * pos_samples) if pos_samples > 0 else 0
        neg_weight = total_samples / (2 * neg_samples) if neg_samples > 0 else 0
        
        # Create weight tensor for each sample
        weights = torch.where(batch.Y == 1, 
                            torch.tensor(pos_weight, device=batch.Y.device),
                            torch.tensor(neg_weight, device=batch.Y.device))
        
        # Calculate weighted BCE loss with properly shaped tensors
        bce_loss = F.binary_cross_entropy_with_logits(
            main_preds, 
            batch.Y.float().squeeze(),
            weight=weights,
            reduction='mean'
        )
        
        # Secondary task loss
        secondary_loss = F.mse_loss(
            secondary_preds.squeeze(),
            batch.secondary_target.squeeze(),
            reduction='mean'
        )
        
        # Log validation metrics
        self.log('val_main_loss', bce_loss, prog_bar=True)
        self.log('val_secondary_loss', secondary_loss, prog_bar=True)
        total_loss = bce_loss - torch.min(self.adversarial_weight * secondary_loss, torch.tensor(150))
        self.log('val_total_loss', total_loss, prog_bar=True)
        return total_loss
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        
        if self.trainer.max_epochs == -1:
            warnings.warn(
                "For infinite training, the number of cooldown epochs is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * self.warmup_epochs * self.trainer.num_training_batches
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * self.trainer.num_training_batches
            
        scheduler = build_NoamLike_LRSched(
            optimizer,
            warmup_steps=self.warmup_epochs * self.trainer.num_training_batches,
            cooldown_steps=cooldown_steps,
            init_lr=self.init_lr,
            max_lr=self.max_lr,
            final_lr=self.final_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }