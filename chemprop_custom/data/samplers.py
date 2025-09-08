from itertools import chain
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler, BatchSampler


class SeededSampler(Sampler):
    """A :class`SeededSampler` is a class for iterating through a dataset in a randomly seeded
    fashion"""

    def __init__(self, N: int, seed: int):
        if seed is None:
            raise ValueError("arg 'seed' was `None`! A SeededSampler must be seeded!")

        self.idxs = np.arange(N)
        self.rg = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        self.rg.shuffle(self.idxs)

        return iter(self.idxs)

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return len(self.idxs)


class ClassBalanceSampler(Sampler):
    """A :class:`ClassBalanceSampler` samples data from a :class:`MolGraphDataset` such that
    positive and negative classes are equally sampled

    Parameters
    ----------
    dataset : MolGraphDataset
        the dataset from which to sample
    seed : int
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """

    def __init__(self, Y: np.ndarray, seed: Optional[int] = None, shuffle: bool = False):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)

        idxs = np.arange(len(Y))
        actives = Y.any(1)

        self.pos_idxs = idxs[actives]
        self.neg_idxs = idxs[~actives]

        self.length = 2 * min(len(self.pos_idxs), len(self.neg_idxs))

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        if self.shuffle:
            self.rg.shuffle(self.pos_idxs)
            self.rg.shuffle(self.neg_idxs)

        return chain(*zip(self.pos_idxs, self.neg_idxs))

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return self.length


class ClassBalanceAndAnchorSampler(Sampler):
    """A :class:`ClassBalanceAndAnchorSampler` samples data from a :class:`MolGraphDataset` such that
    positive and negative classes are equally sampled

    Parameters
    ----------
    dataset : MolGraphDataset
        the dataset from which to sample
    seed : int
        the random seed to use for shuffling (only used when `shuffle` is `True`)
    shuffle : bool, default=False
        whether to shuffle the data during sampling
    """

    def __init__(self, Y: np.ndarray, anchor_idxs: np.ndarray, seed: Optional[int] = None, shuffle: bool = False):
        self.shuffle = shuffle
        self.rg = np.random.default_rng(seed)

        idxs = np.arange(len(Y))
        actives = Y.any(1)

        self.pos_idxs = idxs[actives]
        self.neg_idxs = idxs[~actives]

        self.length = 2 * min(len(self.pos_idxs), len(self.neg_idxs))

    def __iter__(self) -> Iterator[int]:
        """an iterator over indices to sample."""
        if self.shuffle:
            self.rg.shuffle(self.pos_idxs)
            self.rg.shuffle(self.neg_idxs)

        return chain(*zip(self.pos_idxs, self.neg_idxs))

    def __len__(self) -> int:
        """the number of indices that will be sampled."""
        return self.length


class RationaleBatchSampler(BatchSampler):
    """
    Yield batches of indices so that each batch has the same proportion
    of datapoints with rationale_mol (as in the overall dataset).
    """

    def __init__(
        self,
        rationale_mask: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False
    ):
        # rationale_mask: boolean array of length N
        self.rg = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        idxs = np.arange(len(rationale_mask))
        self.rationale_idxs = idxs[rationale_mask]
        self.non_rationale_idxs = idxs[~rationale_mask]

        total = len(idxs)
        p = len(self.rationale_idxs) / total if total > 0 else 0
        # compute how many rationale items per batch (at least 1 if any exist)
        k = int(round(p * batch_size))
        self.k_rationale = max(k, 1) if len(self.rationale_idxs) > 0 else 0
        self.k_non_rationale = batch_size - self.k_rationale

    def __iter__(self):
        pos = list(self.rationale_idxs)
        neg = list(self.non_rationale_idxs)
        if self.shuffle:
            self.rg.shuffle(pos)
            self.rg.shuffle(neg)

        i_pos = i_neg = 0
        n_pos, n_neg = len(pos), len(neg)

        # yield full batches
        while i_pos + self.k_rationale <= n_pos and i_neg + self.k_non_rationale <= n_neg:
            batch = pos[i_pos:i_pos + self.k_rationale] + neg[i_neg:i_neg + self.k_non_rationale]
            if self.shuffle:
                self.rg.shuffle(batch)
            yield batch
            i_pos += self.k_rationale
            i_neg += self.k_non_rationale

        # optionally yield a final smaller batch
        if not self.drop_last and (i_pos < n_pos or i_neg < n_neg):
            batch = pos[i_pos:] + neg[i_neg:]
            if batch:
                if self.shuffle:
                    self.rg.shuffle(batch)
                yield batch

    def __len__(self):
        if self.k_rationale == 0 or self.k_non_rationale == 0:
            # fallback to simple batching if one group is empty
            return int(np.ceil((len(self.rationale_idxs) + len(self.non_rationale_idxs)) / self.batch_size))
        full = min(
            len(self.rationale_idxs) // self.k_rationale,
            len(self.non_rationale_idxs) // self.k_non_rationale
        )
        if not self.drop_last:
            rem_pos = len(self.rationale_idxs) - full * self.k_rationale
            rem_neg = len(self.non_rationale_idxs) - full * self.k_non_rationale
            if rem_pos > 0 or rem_neg > 0:
                return full + 1
        return full


class ClassAndRationaleBatchSampler(BatchSampler):
    """
    Yield batches that are 50/50 positive/negative (class balance) and
    in which the overall fraction of datapoints with rationale_mol
    is constant across batches (rationale balance).
    """

    def __init__(
        self,
        class_mask: np.ndarray,
        rationale_mask: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False
    ):
        # batch_size must be even to split classes evenly
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for class balance.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rg = np.random.default_rng(seed)

        idxs = np.arange(len(class_mask))
        pos = idxs[class_mask]
        neg = idxs[~class_mask]

        # split each class into rationale / non-rationale
        self.pos_r = pos[rationale_mask[pos]]
        self.pos_nr = pos[~rationale_mask[pos]]
        self.neg_r = neg[rationale_mask[neg]]
        self.neg_nr = neg[~rationale_mask[neg]]

        half = batch_size // 2

        # how many total rationals per batch
        total = len(idxs)
        r_total = int(rationale_mask.sum())
        self.k_r = int(round(r_total / total * batch_size)) if total > 0 else 0

        # distribute those rationals between pos/neg in proportion to dataset
        pr_total = len(self.pos_r)
        nr_total = len(self.neg_r)
        self.k_pr = int(round(pr_total / r_total * self.k_r)) if r_total > 0 else 0
        self.k_nr = self.k_r - self.k_pr

        # now each class gets exactly half the batch
        self.k_pos = half
        self.k_neg = half

        # within each class, rationals vs non-rationals
        self.k_pnr = self.k_pos - self.k_pr
        self.k_nnr = self.k_neg - self.k_nr

    def __iter__(self):
        pr = list(self.pos_r)
        pnr = list(self.pos_nr)
        nr = list(self.neg_r)
        nnr = list(self.neg_nr)
        if self.shuffle:
            self.rg.shuffle(pr); self.rg.shuffle(pnr)
            self.rg.shuffle(nr); self.rg.shuffle(nnr)

        i_pr = i_pnr = i_nr = i_nnr = 0
        L_pr, L_pnr = len(pr), len(pnr)
        L_nr, L_nnr = len(nr), len(nnr)

        # yield as many full batches as we can
        while (
            i_pr   + self.k_pr  <= L_pr and
            i_pnr + self.k_pnr <= L_pnr and
            i_nr   + self.k_nr  <= L_nr and
            i_nnr + self.k_nnr <= L_nnr
        ):
            batch = (
                pr[i_pr:i_pr+self.k_pr] +
                pnr[i_pnr:i_pnr+self.k_pnr] +
                nr[i_nr:i_nr+self.k_nr] +
                nnr[i_nnr:i_nnr+self.k_nnr]
            )
            if self.shuffle:
                self.rg.shuffle(batch)
            yield batch
            i_pr  += self.k_pr
            i_pnr += self.k_pnr
            i_nr  += self.k_nr
            i_nnr += self.k_nnr

        # optionally yield a smaller final batch
        if not self.drop_last:
            rem = pr[i_pr:] + pnr[i_pnr:] + nr[i_nr:] + nnr[i_nnr:]
            if rem:
                if self.shuffle:
                    self.rg.shuffle(rem)
                yield rem

    def __len__(self):
        full_batches = min(
            (len(self.pos_r)  // self.k_pr)  if self.k_pr  > 0 else float('inf'),
            (len(self.pos_nr) // self.k_pnr) if self.k_pnr > 0 else float('inf'),
            (len(self.neg_r)  // self.k_nr)  if self.k_nr  > 0 else float('inf'),
            (len(self.neg_nr) // self.k_nnr) if self.k_nnr > 0 else float('inf'),
        )
        if not self.drop_last:
            # check if any leftovers exist
            leftovers = (
                len(self.pos_r)  % self.k_pr  if self.k_pr  > 0 else len(self.pos_r),
                len(self.pos_nr) % self.k_pnr if self.k_pnr > 0 else len(self.pos_nr),
                len(self.neg_r)  % self.k_nr  if self.k_nr  > 0 else len(self.neg_r),
                len(self.neg_nr) % self.k_nnr if self.k_nnr > 0 else len(self.neg_nr),
            )
            if any(leftovers):
                return full_batches + 1
        return full_batches