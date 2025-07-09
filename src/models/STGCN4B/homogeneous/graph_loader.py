from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader

import logging
logger = logging.getLogger(__name__)


class BlockAwareSTGCNDataset(Dataset):
    """
    Dataset of sliding windows over homogeneous feature‐matrix snapshots,
    ensuring windows do not cross block boundaries.

    Each sample is: (X_history, y_target)
        - where X_history is [X_{t-n_his+1}, ..., X_t], a sequence of feature matrices, 
        - and y_target is the target y_t corresponding to the final time step.

    Args:
        blocks: List of Lists, each sublist contains bucket‐indices for one block
        feature_tensor: torch.Tensor of shape (T, R, F)
        target_tensor: torch.Tensor of shape (T, ...) giving the label/target for each bucket
        target_mask_tensor: torch.Tensor of shape (T, ...), binary mask for the target
        target_source_tensor: torch.Tensor of shape (T, ...), original source values for the target (for delta prediction)
        max_target_offset: The maximum future step required by any target. Used to ensure sample validity.
        n_his: history length (number of past buckets)
    """

    def __init__(
        self,
        args,
        blocks: List[List[int]],
        feature_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        target_mask_tensor: torch.Tensor,
        target_source_tensor: torch.Tensor,
        max_target_offset: int,
        n_his: int,
    ):
        self.args = args
        self.feature_tensor = feature_tensor
        self.blocks = blocks
        self.target_tensor = target_tensor
        self.n_his = n_his
        self.target_mask_tensor = target_mask_tensor
        self.target_source_tensor = target_source_tensor
        self.max_target_offset = max_target_offset

        # Precompute valid samples as (block_idx, start_pos)
        self.samples: List[tuple] = []
        for b_idx, block in enumerate(self.blocks):
            L = len(block)
            required_length = n_his + max_target_offset
            if L < required_length:
                continue
            # every start such that [start ... start+n_his-1] fits within block
            for start in range(L - required_length + 1):
                self.samples.append((b_idx, start))
            
        logger.info(f"Initialized Dataset: {len(self.samples)} valid samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        block_idx, start = self.samples[idx]
        block = self.blocks[block_idx]
        
        # History indices
        his_idxs = block[start : start + self.n_his]
        
        # Target index 
        # NOTE: It is a single target index, and corresponds to the last element of the history window.
        target_idx = block[start + self.n_his - 1]
        
        # Indexing operation on the large GPU tensor
        X = self.feature_tensor[his_idxs]
        
        # Gather target values
        y = self.target_tensor[target_idx]

        # Get the mask for the target
        m = self.target_mask_tensor[target_idx]

        # Get the target source values (for delta prediction)
        s = self.target_source_tensor[target_idx]
        
        # Transpose the last two dimensions to align with the model's output (H, R)
        # This is only necessary for tasks with a room dimension.
        if y.ndim == 2: # Apply only to measurement_forecast targets, not consumption/classification
            y = y.transpose(0, 1) # Shape (R, H) -> (H, R)
            m = m.transpose(0, 1) # Shape (R, H) -> (H, R)
            if self.args.prediction_type == "delta": 
                s = s.transpose(0, 1)
                
        return X, y, m, s

def homo_collate(batch):
    """
    Collate function for homogeneous STGCN windows, with optional masking.

    Args:
        batch: List of samples, each is (X, y, target_mask, target_source) where
        - X is a tensor of shape (n_his, R, F)
        - y is a tensor for a single time step (e.g., shape (R,) or a scalar)
        - m (mask for y) is a tensor with the same shape as y
        - s (source of y) is a tensor with the same shape as y
    
    Returns:
        (X_batch, y_batch, mask_batch)
        - X_batch: tensor of shape (batch_size, n_his, R, F)
        - y_batch: tensor of shape (batch_size, ...)
        - m_batch: tensor of shape (batch_size, ...)
        - s_batch: tensor of shape (batch_size, ...)
    """
    X, y, m, s = zip(*batch)

    # Stack into a single batch tensor
    X_batch = torch.stack(X, dim=0)  # Shape: (batch_size, n_his, R, F)

    y_batch = torch.stack(y, dim=0)
    m_batch = torch.stack(m, dim=0)
    s_batch = torch.stack(s, dim=0)
    
    return X_batch, y_batch, m_batch, s_batch

def get_data_loaders(
        args,
        blocks: Dict[int, Dict[str, List[int]]],
        block_size: int,
        feature_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        target_mask_tensor: torch.Tensor,
        target_source_tensor: torch.Tensor,
        max_target_offset: int,
        *, # for safety
        train_block_ids: List[int],
        val_block_ids:   List[int],
        test_block_ids:  List[int]
        ):
    """
    Builds train/val/test DataLoaders from pre-loaded, in-memory data tensors.
        
    VERY IMPORTANT: This function assumes that all blocks in the dataset have the same size.
                    If the blocks have varying sizes, this batch size will not align perfectly 
                    with the boundaries of other blocks. So ensure equal block sizes.
    """    
    # 1) Partition into train/val/test block‐lists
    def _blocks(ids):
        return [ blocks[b]["bucket_indices"] for b in ids ]
    
    train_block_lists: List[List[int]] = _blocks(train_block_ids)
    val_block_lists: List[List[int]]   = _blocks(val_block_ids)  if val_block_ids else []
    test_block_lists: List[List[int]]  = _blocks(test_block_ids) if test_block_ids else []
    
    logger.info(
        f"Using {len(train_block_lists)} train-block(s), "
        f"{len(val_block_lists)} val-block(s), "
        f"{len(test_block_lists)} test-block(s)"
    )
    
    # 2) Construct Datasets
    train_ds = BlockAwareSTGCNDataset(
        args=args,
        blocks=train_block_lists,
        feature_tensor=feature_tensor,
        target_tensor=target_tensor,
        target_mask_tensor=target_mask_tensor,
        target_source_tensor=target_source_tensor,
        max_target_offset=max_target_offset,
        n_his=args.n_his)
    
    # If no validation set is provided, set it to None
    val_ds = None
    if val_block_lists:
        val_ds = BlockAwareSTGCNDataset(
            args=args,
            blocks=val_block_lists,
            feature_tensor=feature_tensor,
            target_tensor=target_tensor,
            target_mask_tensor=target_mask_tensor,
            target_source_tensor=target_source_tensor,
            max_target_offset=max_target_offset,
            n_his=args.n_his)
        
    test_ds = BlockAwareSTGCNDataset(
            args=args,
            blocks=test_block_lists,
            feature_tensor=feature_tensor,
            target_tensor=target_tensor,
            target_mask_tensor=target_mask_tensor,
            target_source_tensor=target_source_tensor,
            max_target_offset=max_target_offset,
            n_his=args.n_his)
        
    # 3) Determine windows_per_block for batch_size
    windows_per_block = block_size - (args.n_his + max_target_offset) + 1
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
    
    logger.info("Block-aware homogeneous data loaders ready (using precomputed blocks).")
    
    return loaders