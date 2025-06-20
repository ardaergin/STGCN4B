import os
import sys
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class BlockAwareSTGCNDataset(Dataset):
    """
    Dataset of sliding windows over homogeneous feature‐matrix snapshots,
    ensuring windows do not cross block boundaries.

    Each sample is:
        ([X_{t-n_his+1}, …, X_t],  y_{t+1:t+n_pred})

    where X_t is the room‐feature matrix (shape R×F) at bucket t,
    and y is the target vector of length n_pred (classification or forecasting).

    Args:
        feature_matrices: Dict[int → torch.Tensor] mapping bucket_idx → (R×F) tensor
        blocks: List of Lists, each sublist contains bucket‐indices for one block
        targets: torch.Tensor of shape (T,) giving label/target for each bucket
        n_his: history length (number of past buckets)
        n_pred: prediction length (number of future buckets)
    """

    def __init__(
        self,
        feature_matrices: Dict[int, torch.Tensor],
        blocks: List[List[int]],
        targets: torch.Tensor,
        n_his: int,
        n_pred: int,
        target_mask: torch.Tensor = None
    ):
        self.feature_matrices = feature_matrices
        self.blocks = blocks
        self.targets = targets
        self.n_his = n_his
        self.n_pred = n_pred
        self.target_mask = target_mask

        # Precompute valid samples as (block_idx, start_pos)
        self.samples: List[tuple] = []
        for b_idx, block in enumerate(self.blocks):
            L = len(block)
            if L < (n_his + n_pred):
                continue
            # every start such that [start ... start+n_his+n_pred−1] fits within block
            for start in range(L - (n_his + n_pred) + 1):
                self.samples.append((b_idx, start))

        logger.info(f"Initialized Dataset: {len(self.samples)} valid samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        block_idx, start = self.samples[idx]
        block = self.blocks[block_idx]

        # Determine history and prediction indices
        his_idxs = block[start : start + self.n_his]
        pred_idxs = block[start + self.n_his : start + self.n_his + self.n_pred]

        # Gather feature matrices for history (list of tensors, each R×F)
        X_list = [self.feature_matrices[t] for t in his_idxs]
        # Gather target values (1D tensor of length n_pred)
        y = self.targets[pred_idxs]
        # Get the mask for the target
        m = self.target_mask[pred_idxs]
        
        return X_list, y, m

def homo_collate(batch):
    """
    Collate function for homogeneous STGCN windows, with optional masking.

    Args:
        batch: List of samples, each
            - (X_list, y, target_mask)
          where
            * X_list is a list of length n_his of tensors, each of shape (R, F)
            * y is a tensor of shape (n_pred, R)
            * target_mask is a tensor of shape (n_pred, R) with 1s where targets are valid

    Returns:
        (X_batch_list, y_batch, mask_batch)
        - X_batch_list: list of length n_his, each element is a tensor of shape
                        (batch_size, R, F)
        - y_batch:     tensor of shape (batch_size, n_pred, R)
        - target_mask_batch:  tensor of shape (batch_size, n_pred, R)
    """
    windows, ys, target_masks = zip(*batch)

    batch_size = len(windows)
    n_his = len(windows[0])

    # Stack target tensors: shape (batch_size, n_pred)
    y_batch = torch.stack(ys, dim=0)
    target_mask_batch = torch.stack(target_masks, dim=0)

    # For each history step t, gather that step across the batch
    X_batch_list: List[torch.Tensor] = []
    for t in range(n_his):
        # windows[i][t] has shape (R, F); stack into (batch_size, R, F)
        step_tensors = [windows[i][t] for i in range(batch_size)]
        X_batch_list.append(torch.stack(step_tensors, dim=0))
    
    return X_batch_list, y_batch, target_mask_batch

def load_data(args,
              blocks: Dict[int, Dict[str, List[int]]],
              feature_matrices,
              targets,
              target_mask,
              *, # for safety
              train_block_ids: List[int],
              val_block_ids:   List[int],
              test_block_ids:  List[int]
              ):
    """
    Builds train/val/test DataLoaders from pre-loaded, in-memory data tensors.

    This function is designed for efficiency, avoiding disk I/O by operating
    on data that is already loaded.
    """    
    # 1) Partition into train/val/test block‐lists
    def _blocks(ids):
        return [ blocks[b]["bucket_indices"] for b in ids ]

    train_block_lists: List[List[int]] = _blocks(train_block_ids)
    val_block_lists: List[List[int]]   = _blocks(val_block_ids)  if val_block_ids   is not None else []
    test_block_lists: List[List[int]]  = _blocks(test_block_ids) if test_block_ids  is not None else []

    logger.info(
        f"Using {len(train_block_lists)} train-block(s), "
        f"{len(val_block_lists)} val-block(s), "
        f"{len(test_block_lists)} test-block(s)"
    )

    # 2) Construct Datasets
    train_ds = BlockAwareSTGCNDataset(
        feature_matrices,
        train_block_lists,
        targets,
        args.n_his,
        args.n_pred,
        target_mask=target_mask
    )
    val_ds = None
    if val_block_lists:
        val_ds = BlockAwareSTGCNDataset(
            feature_matrices,
            val_block_lists,
            targets,
            args.n_his,
            args.n_pred,
            target_mask=target_mask
        )
    test_ds = BlockAwareSTGCNDataset(
        feature_matrices,
        test_block_lists,
        targets,
        args.n_his,
        args.n_pred,
        target_mask=target_mask
    )

    # 3) Determine windows_per_block for batch_size
    #    We can pick the first train block to determine the number of windows
    first_block_len = len(train_ds.blocks[0])
    windows_per_block = first_block_len - (args.n_his + args.n_pred) + 1
    logger.info(f"Calculated batch size: {windows_per_block}")

    # 4) Create DataLoaders (no shuffling; windows are pre‐segmented per block)
    train_loader = DataLoader(
        train_ds,
        batch_size=windows_per_block,
        shuffle=False,
        collate_fn=homo_collate,
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=windows_per_block,
            shuffle=False,
            collate_fn=homo_collate,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=windows_per_block,
        shuffle=False,
        collate_fn=homo_collate,
    )

    # 5) Return everything downstream might need
    loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }

    logger.info("Block‐aware homogeneous data loaders ready (using precomputed blocks).")

    return loaders