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
        mask: torch.Tensor = None
    ):
        self.feature_matrices = feature_matrices
        self.blocks = blocks
        self.targets = targets
        self.n_his = n_his
        self.n_pred = n_pred
        self.mask = mask

        # Precompute valid samples as (block_idx, start_pos)
        self.samples: List[tuple] = []
        for b_idx, block in enumerate(self.blocks):
            L = len(block)
            if L < (n_his + n_pred):
                continue
            # every start such that [start ... start+n_his+n_pred−1] fits within block
            for start in range(L - (n_his + n_pred) + 1):
                self.samples.append((b_idx, start))

        logger.info(
            f"Initialized BlockAwareSTGCNDataset: "
            f"{len(self.blocks)} blocks, {len(self.samples)} valid samples"
        )

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
        # Get the mask
        m = self.mask[pred_idxs]
        
        return X_list, y, m

def homo_collate(batch):
    """
    Collate function for homogeneous STGCN windows, with optional masking.

    Args:
        batch: List of samples, each
            - (X_list, y, mask)
          where
            * X_list is a list of length n_his of tensors, each of shape (R, F)
            * y is a tensor of shape (n_pred, R)
            * mask is a tensor of shape (n_pred, R) with 1s where targets are valid

    Returns:
        (X_batch_list, y_batch, mask_batch)
        - X_batch_list: list of length n_his, each element is a tensor of shape
                        (batch_size, R, F)
        - y_batch:     tensor of shape (batch_size, n_pred, R)
        - mask_batch:  tensor of shape (batch_size, n_pred, R)
    """
    windows, ys, masks = zip(*batch)

    batch_size = len(windows)
    n_his = len(windows[0])

    # Stack target tensors: shape (batch_size, n_pred)
    y_batch = torch.stack(ys, dim=0)
    mask_batch = torch.stack(masks, dim=0)

    # For each history step t, gather that step across the batch
    X_batch_list: List[torch.Tensor] = []
    for t in range(n_his):
        # windows[i][t] has shape (R, F); stack into (batch_size, R, F)
        step_tensors = [windows[i][t] for i in range(batch_size)]
        X_batch_list.append(torch.stack(step_tensors, dim=0))
    
    return X_batch_list, y_batch, mask_batch

def load_data(args):
    """
    Load pre‐processed homogeneous STGCN input and build train/val/test DataLoaders
    with block‐aware temporal windowing.

    Expects `args` to have:
      - data_dir: base path
      - interval: str (e.g., "1h")
      - enable_cuda: bool
      - task_type: 'classification' or 'forecasting'
      - n_his: int
      - n_pred: int

    Returns a dict containing:
      - device
      - time_buckets
      - train_loader, val_loader, test_loader
      - adjacency_matrix
      - dynamic_adjacencies (or None)
      - feature_names, room_uris
      - train_idx, val_idx, test_idx
      - workhour_labels, consumption_values
      - (optionally) any other metadata needed downstream
    """
    # 1) Load saved torch input for homogeneous graph
    if args.task_type == "measurement_forecast":
        fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}_{args.measurement_type}.pt"
    else:
        fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}.pt"
    path = os.path.join(args.data_dir, "processed", fname)
    logger.info(f"Loading homogeneous STGCN input from {path}")
    torch_input = torch.load(path, map_location="cpu")

    # 2) Device setup
    device = (
        torch.device("cuda")
        if args.enable_cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    # 3) Move adjacency matrices to device
    torch_input["adjacency_matrix"] = torch_input["adjacency_matrix"].to(device)
    if torch_input.get("dynamic_adjacencies") is not None:
        for step, mat in torch_input["dynamic_adjacencies"].items():
            torch_input["dynamic_adjacencies"][step] = mat.to(device)

    # 4) Move feature matrices to device
    #    Each entry torch_input["feature_matrices"][t] is a tensor (R×F)
    for t, mat in torch_input["feature_matrices"].items():
        torch_input["feature_matrices"][t] = mat.to(device)

    # 5A) Select targets based on task, move to device
    if args.task_type == "measurement_forecast":
        targets = torch_input["measurement_values"].to(device)  # shape (T, N)
    elif args.task_type == "consumption_forecast":
        targets = torch_input["consumption_values"].to(device) # shape (T, 1)
    elif args.task_type == "workhour_classification":
        targets = torch_input["workhour_labels"].to(device) # shape (T, )

    # 5B) Mask creation
    mask = torch_input.get("measurement_mask") 
    if mask is None:
        logger.info("No measurement mask found. Creating a default mask of all ones.")
        mask = torch.ones_like(targets, dtype=torch.float32)
    mask = mask.to(device)

    # 6) Extract the precomputed blocks dict from torch_input
    full_blocks_dict: Dict[int, Dict[str, List[int]]] = torch_input.get("blocks", None)
    if full_blocks_dict is None:
        raise KeyError(
            "'blocks' not found in torch_input. Expected "
            "torch_input['blocks'] = { block_id: { 'block_type': str, 'bucket_indices': [..] }, ... }"
        )

    # Partition into train/val/test block‐lists
    train_block_lists: List[List[int]] = []
    val_block_lists: List[List[int]] = []
    test_block_lists: List[List[int]] = []

    for block_id, info in full_blocks_dict.items():
        btype = info["block_type"]
        bidxs = info["bucket_indices"]
        if btype == "train":
            train_block_lists.append(bidxs)
        elif btype == "val":
            val_block_lists.append(bidxs)
        elif btype == "test":
            test_block_lists.append(bidxs)
        else:
            raise ValueError(f"Unknown block_type='{btype}' for block_id={block_id}")

    logger.info(
        f"Using {len(train_block_lists)} train-block(s), "
        f"{len(val_block_lists)} val-block(s), "
        f"{len(test_block_lists)} test-block(s)"
    )

    # 7) Construct Datasets
    train_ds = BlockAwareSTGCNDataset(
        torch_input["feature_matrices"],
        train_block_lists,
        targets,
        args.n_his,
        args.n_pred,
        mask=mask
    )
    val_ds = BlockAwareSTGCNDataset(
        torch_input["feature_matrices"],
        val_block_lists,
        targets,
        args.n_his,
        args.n_pred,
        mask=mask
    )
    test_ds = BlockAwareSTGCNDataset(
        torch_input["feature_matrices"],
        test_block_lists,
        targets,
        args.n_his,
        args.n_pred,
        mask=mask
    )

    # 8) Determine windows_per_block for batch_size
    #    We pick the first train block to determine the number of windows
    first_block_len = len(train_ds.blocks[0])
    windows_per_block = first_block_len - (args.n_his + args.n_pred) + 1

    # 9) Create DataLoaders (no shuffling; windows are pre‐segmented per block)
    train_loader = DataLoader(
        train_ds,
        batch_size=windows_per_block,
        shuffle=False,
        collate_fn=homo_collate,
    )
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

    logger.info("Block‐aware homogeneous data loaders ready (using precomputed blocks).")

    # 10) Return everything downstream might need
    input_dict = {
        "device": device,
        "time_buckets": torch_input["time_buckets"],
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        # Adjacency and dynamic adjacency
        "adjacency_matrix": torch_input["adjacency_matrix"],
        "dynamic_adjacencies": torch_input.get("dynamic_adjacencies", None),
        # Feature metadata
        "feature_names": torch_input["feature_names"],
        "room_uris": torch_input["room_uris"],
        # Split indices
        "train_idx": torch_input["train_idx"],
        "val_idx": torch_input["val_idx"],
        "test_idx": torch_input["test_idx"],
        # Targets
        "targets": targets,
        # Number of nodes & features
        "n_nodes": 52,
        "n_features": torch_input["n_features"], # Total features: static + temporal
        "static_feature_count": torch_input["static_feature_count"],
        "temporal_feature_count": torch_input["temporal_feature_count"]
    }

    logger.info("STGCN input preparation complete")

    return input_dict