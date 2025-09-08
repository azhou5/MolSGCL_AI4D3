from dataclasses import InitVar, dataclass, field
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from chemprop_custom.data.datasets import Datum
from chemprop_custom.data.molgraph import MolGraph
from chemprop_custom.data.datapoints import MoleculeDatapoint

@dataclass(repr=False, eq=False, slots=True)
class BatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Sequence[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Sequence[MolGraph]):
        self.__size = len(mgs)
        
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


class TrainingBatch(NamedTuple):
    bmg: BatchMolGraph
    rationale_bmg: BatchMolGraph | None
    neg_rationale_bmg: BatchMolGraph | None
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    secondary_target: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None
    has_rationale_mask: Tensor | None


def collate_batch(batch: Iterable[MoleculeDatapoint]) -> TrainingBatch:
    # Extract all components from each datapoint
    components = [
        (
            d.mg,
            d.rationale_mg,
            d.neg_rationale_mg,
            d.V_d,
            d.x_d,
            d.y,
            d.secondary_target,
            d.weight,
            d.lt_mask,
            d.gt_mask,
            # Keep track of original rationale presence per sample
            (d.rationale_mg is not None) and (d.neg_rationale_mg is not None)
        ) for d in batch
    ]

    # Unzip the components
    (mgs, rationale_mgs, neg_rationale_mgs, V_ds, x_ds, ys, secondary_targets, weights,
     lt_masks, gt_masks, has_both_rationales_flags) = zip(*components) # Unpack the new flag list

    # Build BatchMolGraph for main molecules
    bmg = BatchMolGraph(mgs)

    # Filter and build BatchMolGraph for rationales
    pos_mgs = [mg for mg in rationale_mgs if mg is not None]
    rationale_bmg = BatchMolGraph(pos_mgs) if pos_mgs else None
    neg_mgs = [mg for mg in neg_rationale_mgs if mg is not None]
    neg_rationale_bmg = BatchMolGraph(neg_mgs) if neg_mgs else None

    # Other fields
    V_d = None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float()
    X_d = None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float()
    Y = None if ys[0] is None else torch.from_numpy(np.array(ys)).float()
    secondary_target = None if secondary_targets[0] is None else torch.from_numpy(np.array(secondary_targets)).float()
    w = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    lt_mask = None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks))
    gt_mask = None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks))

    # Convert the per-sample flags into the mask tensor
    has_rationale_mask = torch.tensor(has_both_rationales_flags, dtype=torch.bool)

    return TrainingBatch(
        bmg,
        rationale_bmg,
        neg_rationale_bmg,
        V_d,
        X_d,
        Y,
        secondary_target,
        w,
        lt_mask,
        gt_mask,
        has_rationale_mask # Pass the computed mask
    )


class MulticomponentTrainingBatch(NamedTuple):
    bmgs: list[BatchMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    secondary_target: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_multicomponent(batches: Iterable[Iterable[Datum]]) -> MulticomponentTrainingBatch:
    tbs = [collate_batch(batch) for batch in zip(*batches)]

    return MulticomponentTrainingBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].secondary_target,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )
