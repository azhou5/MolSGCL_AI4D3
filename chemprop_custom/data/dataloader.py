import warnings

from torch.utils.data import DataLoader
import numpy as np

from chemprop_custom.data.collate import collate_batch, collate_multicomponent
from chemprop_custom.data.datasets import MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop_custom.data.samplers import (
    ClassBalanceSampler,
    SeededSampler,
    RationaleBatchSampler,
    ClassAndRationaleBatchSampler,
)


def build_dataloader(
    dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    rationale_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,
    **kwargs,
):
    """Return a :obj:`~torch.utils.data.DataLoader` for :class:`MolGraphDataset`\s

    Parameters
    ----------
    dataset : MoleculeDataset | ReactionDataset | MulticomponentDataset
        The dataset containing the molecules or reactions to load.
    batch_size : int, default=64
        the batch size to load.
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules). Class balance is only available for single task classification datasets. Set
        shuffle to True in order to get a random subset of the larger class.
    rationale_balance : bool, default=False
        Whether to perform rationale balancing (i.e., use an equal number of datapoints with a non-null rationale_mol).
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`).
    shuffle : bool, default=False
        whether to shuffle the data during sampling.
    """

    # choose collate fn
    if isinstance(dataset, MulticomponentDataset):
        collate_fn = collate_multicomponent
    else:
        collate_fn = collate_batch

    # avoid last batch of size 1
    if len(dataset) % batch_size == 1:
        warnings.warn(
            f"Dropping last batch of size 1 to avoid issues with batch normalization \
(dataset size = {len(dataset)}, batch_size = {batch_size})"
        )
        drop_last = True
    else:
        drop_last = False

    # Combined class + rationale balance
    if class_balance and rationale_balance:
        dps = dataset

        class_mask     = np.array([bool(dp.y is not None and dp.y.any()) for dp in dps])
        rationale_mask = np.array([dp.rationale_mg is not None for dp in dps])

        batch_sampler = ClassAndRationaleBatchSampler(
            class_mask, rationale_mask,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs,
        )

    # fallback to single-balance or plain samplers
    if class_balance:
        sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
    elif rationale_balance:
        try:
            dps = dataset.datapoints
        except AttributeError:
            raise AttributeError(
                "To use rationale_balance, Dataset must expose `dataset.datapoints`."
            )
        mask = np.array([dp.rationale_mol is not None for dp in dps])
        sampler = RationaleBatchSampler(mask, batch_size, shuffle, seed, drop_last)
    elif shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )
