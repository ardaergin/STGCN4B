from typing import Dict, List, Any, Tuple, Literal
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, HeteroData
from abc import ABC, abstractmethod

import logging; logger = logging.getLogger(__name__)


class STGCNDataset(Dataset, ABC):
    """
    Abstract base class for creating time-series datasets for STGCN models.
    Importantly, this Dataset class takes care of the stratified nature (i.e., blocks) of the data.
    """
    def __init__(
            self,
            args: Any,
            blocks: List[List[int]],
            target_tensor: torch.Tensor,
            target_mask_tensor: torch.Tensor,
            target_source_tensor: torch.Tensor,
            max_horizon: int,
            n_his: int,
            padding_strategy: Literal["zero", "replication"] = "zero",
    ):
        super().__init__()
        self.args = args
        self.blocks = blocks
        self.target_tensor = target_tensor
        self.target_mask_tensor = target_mask_tensor
        self.target_source_tensor = target_source_tensor
        self.n_his = n_his
        self.max_horizon = max_horizon
        self.padding_strategy = padding_strategy

        # Precompute valid sample indices as (block_index, end_position_in_block)
        self.samples: List[Tuple[int, int]] = []
        for b_idx, block in enumerate(self.blocks):
            # A sample is valid if its target and the max forecast horizon fit within the block.
            # The last possible end_pos for a history window must be `max_horizon` steps from the end of the block.
            num_timesteps = len(block)
            if num_timesteps < max_horizon:
                continue
            
            # The last valid time step to generate a target for is `num_timesteps - max_horizon`.
            # We can create a sample for every possible end position up to this point.
            for end_pos_in_block in range(num_timesteps - max_horizon + 1):
                self.samples.append((b_idx, end_pos_in_block))

        logger.info(f"Initialized Dataset with {len(self.samples)} valid samples across {len(blocks)} blocks.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
            self, 
            idx: int
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample, which includes the history (X) and target (y, m, s).
        Must be implemented by subclasses.
        """
        block_idx, end_pos_in_block = self.samples[idx]
        block = self.blocks[block_idx]

        # Determine history indices ad required padding
        start_pos_in_block = end_pos_in_block - self.n_his + 1
        num_padding = -start_pos_in_block if start_pos_in_block < 0 else 0

        if num_padding > 0:
            history_indices = block[0 : end_pos_in_block + 1]
        else:
            history_indices = block[start_pos_in_block : end_pos_in_block + 1]

        # 2. Delegate history retrieval to the concrete subclass
        X = self._get_history(history_indices, num_padding)

        # 3. Get target-related tensors
        y, m, s = self._get_target_data(end_pos_in_block, block)
        
        return X, y, m, s
    
    @abstractmethod
    def _get_history(
            self, 
            history_indices:    List[int], 
            num_padding:        int
    ) -> Any:
        """
        Retrieves and pads history features (X). Must be implemented by subclasses.
        
        Args:
            history_indices: The list of time indices for which to fetch data.
            num_padding: The number of padding elements to prepend.
        
        Returns:
            The history features (X), which can be a Tensor or a list of graphs.
        """
        pass

    def _get_target_data(
            self, 
            end_pos_in_block:   int, 
            block:              List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper to fetch target, mask, and source tensors for a given time index."""
        target_idx = block[end_pos_in_block]
        
        y = self.target_tensor[target_idx]
        m = self.target_mask_tensor[target_idx]
        s = self.target_source_tensor[target_idx]
        
        # Transpose the last two dimensions to align with model's output (H, R) vs (R, H)
        # This is only necessary for tasks with a room dimension (measurement forecast).
        if y.ndim == 2:
            y = y.transpose(0, 1)
            m = m.transpose(0, 1)
            if self.args.prediction_type == "delta":
                s = s.transpose(0, 1)
        
        return y, m, s


class Homogeneous(STGCNDataset):
    """Dataset for homogeneous graphs, where features are in a single (T, R, F) tensor."""
    def __init__(
            self, 
            data: torch.Tensor, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.data = data
        logger.info("Initialized Homogeneous STGCNDataset.")

    def _get_history(
            self, 
            history_indices: List[int], 
            num_padding: int
    ) -> torch.Tensor:
        # Retrieve the history features from the data tensor
        actual_X = self.data[history_indices]

        # If padding is needed, do replication padding
        if num_padding > 0:
            if self.padding_strategy == "replication":
                padding_tensor = actual_X[0].unsqueeze(0).repeat(num_padding, 1, 1)
            elif self.padding_strategy == "zero":
                padding_tensor = torch.zeros(
                    (num_padding, actual_X.shape[1], actual_X.shape[2]),
                    dtype=actual_X.dtype,
                    device=actual_X.device
                )
            else:
                raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")
            X_features = torch.cat([padding_tensor, actual_X], dim=0)
        else:
            X_features = actual_X
            
        # Add padding mask as a feature
        padding_mask = torch.zeros(
            (self.n_his, 1), 
            dtype=actual_X.dtype, 
            device=actual_X.device
        )
        padding_mask[:num_padding] = 1.0
        num_rooms = X_features.shape[1]
        padding_mask_feature = padding_mask.unsqueeze(1).repeat(1, num_rooms, 1)
        
        # This returns a tensor of shape (Time, Rooms, Features)
        X = torch.cat([X_features, padding_mask_feature], dim=2)
        
        # Permute to the model's expected format (F, T, R)
        X = X.permute(2, 0, 1)
        
        return X



class Heterogeneous(STGCNDataset):
    """
    Dataset for heterogeneous graphs, using a dictionary of temporal graph snapshots.
    
    **Important notes**:
    - For padding, we are not deep copying. Because the same snapshot object is reused across
    multiple padded positions within the same sample and across batches, any
    in-place modification of its .x, .edge_attr, or other tensors will
    affect every reference. As long as the snapshot objects are treated as
    read-only (and most pipelines do), this is entirely safe and maximally efficient.
    - If any node type's count can change across time (e.g. sensors drop out), 
    the collate will torch.stack different sizes and crash. But this is not an issue for us,
    since the nodes are fixed in our datasets.
    """
    def __init__(
            self, 
            data: Dict[int, HeteroData], 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.temporal_graphs = data
        self._zero_graph_cache: HeteroData | None = None
        logger.info("Initialized Heterogeneous STGCNDataset.")
    
    @staticmethod
    def _make_zero_graph(template: HeteroData) -> HeteroData:
        """Create a graph with identical topology but zeroed node features."""
        z = HeteroData()
        # Node features
        for ntype, store in template.node_items():
            z[ntype].x = torch.zeros_like(store["x"], device=store["x"].device)
        # Edge indices (and attrs if they exist)
        for etype, store in template.edge_items():
            z[etype].edge_index = store["edge_index"]
            if "edge_attr" in store:
                z[etype].edge_attr = torch.zeros_like(store["edge_attr"], device=store["edge_attr"].device)
        return z
    
    def _get_history(
            self, 
            history_indices: List[int], 
            num_padding: int
    ) -> List[HeteroData]:
        actual_graphs = [self.temporal_graphs[i] for i in history_indices]
        
        if num_padding == 0:
            return actual_graphs

        first_graph = actual_graphs[0]

        if self.padding_strategy == "replication":
            padding_graphs = [first_graph] * num_padding
        elif self.padding_strategy == "zero":
            # build once per dataset instance, then reuse
            if self._zero_graph_cache is None:
                self._zero_graph_cache = self._make_zero_graph(first_graph)
            padding_graphs = [self._zero_graph_cache] * num_padding
        else:
            raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")

        return padding_graphs + actual_graphs

# ==================================
# Collate Functions
# ==================================

def homo_collate(
        batch: List[Tuple]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates a batch of homogeneous data samples."""
    X, y, m, s = zip(*batch)
    X_batch = torch.stack(X, dim=0)
    y_batch = torch.stack(y, dim=0)
    m_batch = torch.stack(m, dim=0)
    s_batch = torch.stack(s, dim=0)
    return X_batch, y_batch, m_batch, s_batch


def hetero_collate(
    batch: List[
        Tuple[
            List[HeteroData],      # X : length‑n_his list of HeteroData
            torch.Tensor,          # y
            torch.Tensor,          # m
            torch.Tensor           # s
        ]
    ]
) -> Tuple[
        Dict[str, Any], 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ]:
    """
    (1) Transposes the outer list to group snapshots by time,
    (2) stacks node features so the network receives dense tensors.
    
    Returns
    -------
    x_pack   : {"features": Dict[str, Tensor]  # (B, T, N, C)}
    y_batch  : Tensor            (B, ...)
    m_batch  : Tensor            (B, ...)
    s_batch  : Tensor            (B, ...)
    """
    # Unpack batch
    x_lists, ys, ms, ss = zip(*batch)   # |x_lists| = B, each is list[HeteroData]
    B        = len(x_lists)
    T        = len(x_lists[0])          # n_his
    
    # Sanity check: all samples same history length
    assert all(len(x) == T for x in x_lists), "Inconsistent history length in batch"
    
    # Group graphs by timestep
    snaps_by_t = list(zip(*x_lists))    # len = T, each item is tuple[HeteroData] of size B
    
    # Build dense tensors per node type
    node_types = snaps_by_t[0][0].node_types
    feat_dict = {}

    for ntype in node_types:
        per_sample = []
        for b in range(B):
            feats = [snaps_by_t[t][b][ntype].x for t in range(T)]   # (T, N, C)
            per_sample.append(torch.stack(feats, 0))
        # Stack to get (B, T, N, C)
        tensor_b_t_n_c = torch.stack(per_sample, 0)
        
        # Permute to (B, C, T, N) to mirror the homogeneous format
        feat_dict[ntype] = tensor_b_t_n_c.permute(0, 3, 1, 2)
    
    # last_snaps = [x_list[-1] for x_list in x_lists]
    # last_batch = Batch.from_data_list(last_snaps)
    # edge_dict = {}
    # for edge_type in last_batch.edge_types:
    #     edge_store = last_batch[edge_type]
    #     edge_dict[edge_type] = {
    #         'index': edge_store.edge_index,
    #         'weight': getattr(edge_store, 'edge_attr', None)
    #     }

    # Stack the features and edges into a single dictionary
    x_pack = {
        "features": feat_dict, 
        # "edges": edge_dict
    }
    
    # Stack y, m, s (same as homogeneous collate)
    y_batch = torch.stack(ys, dim=0)
    m_batch = torch.stack(ms, dim=0)
    s_batch = torch.stack(ss, dim=0)

    return x_pack, y_batch, m_batch, s_batch


# ==================================
# DataLoader Factory
# ==================================

def seed_worker(worker_id: int):
    """Sets a unique random seed for each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loaders(
        args:                   Any,
        seed:                   int,
        blocks:                 Dict[int, Dict[str, List[int]]],
        target_tensor:          torch.Tensor,
        train_mask_tensor:      torch.Tensor,
        eval_mask_tensor:       torch.Tensor,
        target_source_tensor:   torch.Tensor,
        max_horizon:            int,
        *,
        train_block_ids:        List[int],
        val_block_ids:          List[int],
        test_block_ids:         List[int],
        # Graph-type specific arguments:
        graph_type: str,
        feature_data:           torch.Tensor | Dict[int, HeteroData],
) -> Dict[str, DataLoader]:
    """
    Factory function to build train/val/test DataLoaders for either
    homogeneous or heterogeneous graph data.
    """
    # 1. Partition block indices for each split
    def _get_block_lists(ids: List[int]) -> List[List[int]]:
        return [blocks[b_id]["bucket_indices"] for b_id in ids]
    
    train_blocks    = _get_block_lists(train_block_ids)
    val_blocks      = _get_block_lists(val_block_ids)   if val_block_ids else []
    test_blocks     = _get_block_lists(test_block_ids)  if test_block_ids else []

    logger.info(
        f"Using {len(train_blocks)} train-block(s), "
        f"{len(val_blocks)} val-block(s), "
        f"{len(test_blocks)} test-block(s)"
    )
    
    # 2. Select Dataset class, features, and collate function based on graph_type
    if graph_type == "homogeneous":
        DatasetClass        = Homogeneous
        collate_fn          = homo_collate
    elif graph_type == "heterogeneous":
        DatasetClass        = Heterogeneous
        collate_fn          = hetero_collate
    else:
        raise ValueError(f"Unsupported graph_type: '{graph_type}'")

    # 3. Instantiate Datasets for each split
    common_params = {
        "args":                     args,
        "target_tensor":            target_tensor,
        "target_source_tensor":     target_source_tensor,
        "max_horizon":              max_horizon,
        "n_his":                    args.n_his,
        "padding_strategy":         args.padding_strategy,
    }
    
    train_ds = DatasetClass(
        blocks=train_blocks, 
        data=feature_data, 
        target_mask_tensor=train_mask_tensor, 
        **common_params
    )
    
    val_ds = None
    if val_blocks:
        val_ds = DatasetClass(
            blocks=val_blocks, 
            data=feature_data, 
            target_mask_tensor=eval_mask_tensor,
            **common_params
        )
        
    test_ds = DatasetClass(
        blocks=test_blocks, 
        data=feature_data, 
        target_mask_tensor=eval_mask_tensor,
        **common_params
    )

    # 4. Create DataLoaders
    generator = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(
        dataset         = train_ds,
        batch_size      = args.batch_size,
        shuffle         = True,
        collate_fn      = collate_fn,
        num_workers     = args.num_dataloader_workers,
        pin_memory      = True,
        # Train-specific:
        drop_last       = args.drop_last_batch,
        generator       = generator,
        worker_init_fn  = seed_worker,
    )
    
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            dataset         = val_ds,
            batch_size      = args.batch_size,
            shuffle         = False,
            collate_fn      = collate_fn,
            num_workers     = args.num_dataloader_workers,
            pin_memory      = True,
        )
    
    test_loader = DataLoader(
        dataset         = test_ds,
        batch_size      = args.batch_size,
        shuffle         = False,
        collate_fn      = collate_fn,
        num_workers     = args.num_dataloader_workers,
        pin_memory      = True,
    )
    
    logger.info(f"Created {graph_type} data loaders successfully.")
    return {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}