import os
import sys
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoDataLoader

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class BlockAwareSTGCNDataset(Dataset):
    """
    Dataset of sliding windows over hetero-graph snapshots,
    ensuring windows do not cross block boundaries.

    Each sample is ([G_{t-n_his+1}, …, G_t], y_{t+1:t+n_pred}).
    """

    def __init__(
        self,
        temporal_graphs: Dict[int, torch.nn.Module],
        indices: List[int],
        n_his: int,
        n_pred: int,
        targets: torch.Tensor
    ):
        self.graphs = temporal_graphs
        self.n_his = n_his
        self.n_pred = n_pred
        self.targets = targets
        
        # Build blocks of consecutive indices
        self.blocks = self._make_blocks(sorted(indices))
        # Precompute valid samples as (block_idx, start_pos)
        self.samples = []
        for b_idx, block in enumerate(self.blocks):
            L = len(block)
            if L < (n_his + n_pred):
                continue
            for start in range(L - (n_his + n_pred) + 1):
                self.samples.append((b_idx, start))

        logger.info(
            f"Initialized BlockAwareSTGCNDataset: {len(self.blocks)} blocks, "
            f"{len(self.samples)} valid samples"
        )

    def _make_blocks(self, idxs: List[int]) -> List[List[int]]:
        """Group sorted time indices into consecutive blocks."""
        if not idxs:
            return []
        blocks = []
        cur = [idxs[0]]
        for prev, curr in zip(idxs, idxs[1:]):
            if curr == prev + 1:
                cur.append(curr)
            else:
                blocks.append(cur)
                cur = [curr]
        blocks.append(cur)
        return blocks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        block_idx, start = self.samples[idx]
        block = self.blocks[block_idx]

        # History and prediction windows
        his_idxs = block[start : start + self.n_his]
        pred_idxs = block[start + self.n_his : start + self.n_his + self.n_pred]

        graphs = [self.graphs[t] for t in his_idxs]
        y = self.targets[pred_idxs]
        return graphs, y


def hetero_collate(batch):
    """
    Collate function for heterogeneous STGCN windows.

    Args:
        batch: List of tuples ([HeteroData graphs], y)
    Returns:
        List[Batch] of length n_his, and Tensor y of shape [batch_size, n_pred]
    """
    windows, ys = zip(*batch)
    batch_size = len(windows)
    n_his = len(windows[0])

    # Stack target tensors: [batch_size, n_pred]
    y = torch.stack(ys, dim=0)

    batched_graphs = []
    for t in range(n_his):
        slice_t = [windows[i][t] for i in range(batch_size)]
        batched_graphs.append(Batch.from_data_list(slice_t))

    return batched_graphs, y


def load_and_split_data(args):
    """
    Load pre-processed hetero-STGCN input and build train/val/test DataLoaders
    with block-aware temporal windowing.

    Expects 'args' to have:
      - data_dir: base path
      - adjacency_type: str
      - interval: str
      - enable_cuda: bool
      - task_type: 'classification' or 'forecasting'
      - n_his: int
      - n_pred: int
    """
    # 1) Load input
    fname = f"torch_input_weighted_1h_heterogenous.pt"
    path = os.path.join(args.data_dir, "processed", fname)
    logger.info(f"Loading torch input from {path}")
    torch_input = torch.load(path, map_location="cpu")

    # 2) Device setup
    device = (
        torch.device("cuda")
        if args.enable_cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    # 3) Move adjacency matrices
    for key in (
        "horizontal_adjacency_matrix",
        "vertical_adj_matrix",
        "combined_adjacency_matrix",
    ):
        torch_input[key] = torch_input[key].to(device)
    if torch_input.get("dynamic_adjacencies"):
        for step, mat in torch_input["dynamic_adjacencies"].items():
            torch_input["dynamic_adjacencies"][step] = mat.to(device)

    # 4) Move graphs & targets
    torch_input["base_graph"] = torch_input["base_graph"].to(device)
    for t, G in torch_input["temporal_graphs"].items():
        torch_input["temporal_graphs"][t] = G.to(device)

    torch_input["workhour_labels"] = torch_input["workhour_labels"].to(device)
    torch_input["consumption_values"] = torch_input["consumption_values"].to(device)

    # 5) Splits
    train_idx = torch_input["train_idx"]
    val_idx   = torch_input["val_idx"]
    test_idx  = torch_input["test_idx"]
    logger.info(
        f"Splits → Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    # 6) Dataset construction
    targets = (
        torch_input["workhour_labels"]
        if args.task_type == "classification"
        else torch_input["consumption_values"]
    )

    train_ds = BlockAwareSTGCNDataset(
        torch_input["temporal_graphs"], train_idx,
        args.n_his, args.n_pred, targets
    )
    val_ds = BlockAwareSTGCNDataset(
        torch_input["temporal_graphs"], val_idx,
        args.n_his, args.n_pred, targets
    )
    test_ds = BlockAwareSTGCNDataset(
        torch_input["temporal_graphs"], test_idx,
        args.n_his, args.n_pred, targets
    )

    # 7) DataLoader batch_size = windows per first train block
    if train_ds.blocks:
        first_block_len = len(train_ds.blocks[0])
        windows_per_block = first_block_len - (args.n_his + args.n_pred) + 1
    else:
        windows_per_block = args.batch_size

    train_loader = GeoDataLoader(
        train_ds,
        batch_size=windows_per_block,
        shuffle=False,
        collate_fn=hetero_collate,
    )
    val_loader = GeoDataLoader(
        val_ds,
        batch_size=windows_per_block,
        shuffle=False,
        collate_fn=hetero_collate,
    )
    test_loader = GeoDataLoader(
        test_ds,
        batch_size=windows_per_block,
        shuffle=False,
        collate_fn=hetero_collate,
    )

    logger.info("Block-aware data loaders ready.")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "horizontal_adjacency_matrix": torch_input["horizontal_adjacency_matrix"],
        "vertical_adjacency_matrix":   torch_input["vertical_adj_matrix"],
        "combined_adjacency_matrix":   torch_input["combined_adjacency_matrix"],
        "dynamic_adjacencies":         torch_input.get("dynamic_adjacencies"),
        "base_graph":                  torch_input["base_graph"],
        "time_buckets":                torch_input["time_buckets"],
        "device":                      device,
    }
