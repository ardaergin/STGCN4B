from typing import Dict, List, Any, Literal
import random
import numpy as np
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
        max_horizon: The maximum future step required by any target. Used to ensure sample validity.
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
        max_horizon: int,
        n_his: int,
        padding_strategy: Literal["zero", "replication"] = "zero",
    ):
        self.args = args
        self.feature_tensor = feature_tensor
        self.padding_strategy = padding_strategy
        self.blocks = blocks
        self.target_tensor = target_tensor
        self.n_his = n_his
        self.target_mask_tensor = target_mask_tensor
        self.target_source_tensor = target_source_tensor
        self.max_horizon = max_horizon

        # Precompute valid samples as (block_idx, start_pos)
        self.samples: List[tuple] = []
        for b_idx, block in enumerate(self.blocks):
            L = len(block)
            # A sample is valid if its target and max_horizon fit within the block.
            # The last possible end_pos must be `max_horizon` steps from the end.
            if L < max_horizon:
                continue
            # Create a sample for every possible end_pos, starting from the very beginning.
            for end_pos_in_block in range(L - max_horizon + 1):
                self.samples.append((b_idx, end_pos_in_block))
            
        logger.info(f"Initialized Dataset: {len(self.samples)} valid samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        block_idx, end_pos = self.samples[idx]
        block = self.blocks[block_idx]
        target_idx = block[end_pos]
        start_pos = end_pos - self.n_his + 1
        
        ### Padding logic ###

        # Initialize the padding mask, shape (n_his, 1)
        padding_mask = torch.zeros((self.n_his, 1))
        
        if start_pos < 0:
            num_padding = -start_pos
            actual_his_idxs = block[0 : end_pos + 1]
            actual_X = self.feature_tensor[actual_his_idxs]
            _, num_rooms, num_features = self.feature_tensor.shape
            
            # Padding
            if self.padding_strategy == "replication":
                first_feature_vector = actual_X[0, :, :]
                padding_tensor = first_feature_vector.unsqueeze(0).repeat(num_padding, 1, 1)
            
            elif self.padding_strategy == "zero":
                padding_tensor = torch.zeros(
                    (num_padding, num_rooms, num_features),
                    dtype=self.feature_tensor.dtype
                )
            
            X_features = torch.cat([padding_tensor, actual_X], dim=0)
            
            # Mark the padded steps in the mask (e.g., with a 1)
            padding_mask[:num_padding, :] = 1.0
        
        else:
            # No padding, so the mask remains all zeros.
            his_idxs = block[start_pos : end_pos + 1]
            X_features = self.feature_tensor[his_idxs]
        
        # Concatenating the padding mask as a new feature
        # X_features shape: (n_his, R, F)
        # We need to broadcast the mask to (n_his, R, 1) to concatenate
        R = X_features.shape[1]
        padding_mask_feature = padding_mask.unsqueeze(1).repeat(1, R, 1) # Shape: (n_his, R, 1)
        
        # X becomes the final input tensor with F+1 features
        X = torch.cat([X_features, padding_mask_feature], dim=2)
        
        ### End of Padding logic ###
        
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

def seed_worker(worker_id: int):
    """
    Helper function to set a unique seed for each DataLoader worker.

    The `worker_id` parameter is unused but required by the `worker_init_fn` interface.
    PyTorch's DataLoader automatically seeds each worker with `base_seed + worker_id`,
    which is retrieved via `torch.initial_seed()`. This function uses that unique
    seed to seed other libraries like NumPy and Python's random module.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_loaders(
        args: Any,
        seed: int,
        blocks: Dict[int, Dict[str, List[int]]],
        block_size: int,
        feature_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        target_mask_tensor: torch.Tensor,
        target_source_tensor: torch.Tensor,
        max_horizon: int,
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
        max_horizon=max_horizon,
        n_his=args.n_his,
        padding_strategy=args.padding_strategy)
    
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
            max_horizon=max_horizon,
            n_his=args.n_his,
            padding_strategy=args.padding_strategy)
        
    test_ds = BlockAwareSTGCNDataset(
            args=args,
            blocks=test_block_lists,
            feature_tensor=feature_tensor,
            target_tensor=target_tensor,
            target_mask_tensor=target_mask_tensor,
            target_source_tensor=target_source_tensor,
            max_horizon=max_horizon,
            n_his=args.n_his,
            padding_strategy=args.padding_strategy)
        
    # 3) Logging related batch size & windows per block
    windows_per_block = block_size - max_horizon + 1
    logger.info(f"Windows per block: {windows_per_block}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create a generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 4) Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=homo_collate,
        drop_last=args.drop_last_batch,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=args.num_dataloader_workers,
        pin_memory=True
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=homo_collate,
            drop_last=False,
            num_workers=args.num_dataloader_workers,
            pin_memory=True
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=homo_collate,
        drop_last=False,
        num_workers=args.num_dataloader_workers,
        pin_memory=True
    )
    
    # 5) Return everything downstream might need
    loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
    
    logger.info("Block-aware homogeneous data loaders ready.")
    
    return loaders