import os
import json
import numpy as np
from typing import List, Tuple, Generator, Dict, TypedDict


import logging
logger = logging.getLogger(__name__)


class CVFold(TypedDict):
    """A dictionary representing a single fold in the cross-validation."""
    train_block_ids: List[int]
    val_block_ids: List[int]

class StratifiedBlockSplitter:
    """
    A helper class to perform stratified, random block-based splitting for time series data.
    The class has cross-validation support as well as predefined train-val-test split option.

    This class groups sequential blocks into temporal strata and then performs
    sampling within these strata. This ensures that temporal periods are
    proportionally represented in all sets, which is crucial for seasonal or cyclical data.

    Example:
        >>> # Assume `all_blocks` is a pre-existing dict and `your_data_array` is a NumPy array.
        >>>
        >>> # 1. Initialize the splitter
        >>> splitter = StratifiedBlockSplitter(blocks=all_blocks, stratum_size=5)
        >>>
        >>> # 2. Create the main train/test split and get test indices
        >>> splitter.get_train_test_split()
        >>> test_indices = splitter.get_test_idx()
        >>>
        >>> # 3. Create CV folds and loop through them to get train/validation indices
        >>> splitter.get_cv_splits()
        >>>
        >>> # 4. Loop through the CV folds using the generators
        >>> train_gen = splitter.get_train_idx()
        >>> val_gen = splitter.get_val_idx()
        >>>
        >>> for i, (train_indices, val_indices) in enumerate(zip(train_gen, val_gen)):
        ...     # train_data = your_data_array[train_indices]
        ...     # val_data = your_data_array[val_indices]
    """
    def __init__(self, output_dir: str, blocks: Dict[int, dict],
                 stratum_size: int = 5, seed: int = 2658918):
        """
        Initializes the splitter with a pre-built dictionary of blocks.
        
        Args:
            blocks: A dictionary where keys are integer block IDs and values are
                    dictionaries containing block information. The splitter relies
                    on the keys being sorted to maintain temporal order.
                    Example:   {0: {'bucket_indices': [0, 1, ..., 143]}, 
                                1: {'bucket_indices': [144, 145, ..., 287]},
                                2: {...}}
            stratum_size:  This corresponds to the number of blocks in each strata.
                            Always, the number of test blocks in each strata will be one.
                            And, always, for each CV split, there will be only one validation block in each strata.
                            Hence, this parameter impacts both the k in CV as well as test size.
                            Let's say "5" is specified. Then, the strata ratio is 3:1:1. 
                            Thus, test size is 20%, and k=4 for CV.
            seed: Seed for reproducibility.
            output_dir: JSON split files will be saved here.
        """

        # Validating inputs
        if not isinstance(blocks, dict) or not blocks:
            raise ValueError("`blocks` must be a non-empty dict.")
        if not isinstance(stratum_size, int) or stratum_size < 3:
            raise ValueError("`stratum_size` must be an integer >= 3.")
        if len(blocks) < stratum_size:
            raise ValueError(
                f"Need at least {stratum_size} blocks to form one full stratum;"
                f" got {len(blocks)}."
            )
        logger.info(f"StratifiedBlockSplitter initialized with stratum_size {stratum_size}.")

        # Blocks
        self.blocks = blocks
        self.block_ids = np.array(sorted(list(self.blocks.keys())))
        logger.info(f"StratifiedBlockSplitter initialized with {len(self.block_ids)} blocks.")

        # Seed
        self.seed = seed
        seed_seq = np.random.SeedSequence(self.seed)
        child_seeds = seed_seq.spawn(2)
        self.train_test_rng = np.random.default_rng(child_seeds[0])
        self.validation_rng = np.random.default_rng(child_seeds[1])

        # Strata
        self.stratum_size = stratum_size 
        self.n_train_in_strata = stratum_size - 2 # i.e., - 1 test block - 1 val block

        positions = np.arange(len(self.block_ids))
        strata_labels = positions // stratum_size
        unique_strata = np.unique(strata_labels)
        self.strata: Dict[int, np.ndarray] = {
            strata: self.block_ids[strata_labels == strata] 
            for strata in unique_strata
        }
        logger.info(f"Created {len(self.strata)} strata for {len(self.block_ids)} blocks.")

        # K-fold
        self.n_splits = stratum_size - 1

        # Initialize attributes to store the splits
        self.train_block_ids: List[int] = None
        self.test_block_ids: List[int] = None
        self.CV_splits: List[CVFold] = None

        # For saving
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_filename = os.path.join(self.output_dir, f"{self.seed}.json")
        
    def _save_block_split(self, split_type: str,
                        train_ids: List[int], val_ids: List[int], test_ids: List[int]
                        ) -> None:
        """
        Internal helper: save or append splits in one JSON file.
        - 'train_test' overwrites/creates base structure.
        - 'cv' appends under 'cv_folds'.
        """
        data: Dict[str, any]

        if split_type == 'train_test':
            data = {
                'seed': self.seed,
                'train_blocks': [int(i) for i in train_ids],
                'test_blocks': [int(i) for i in test_ids]
            }
            with open(self.save_filename, 'w') as fp:
                json.dump(data, fp, indent=2)
            logger.info(f"Wrote train-test split to {self.save_filename}")

        elif split_type == 'cv':
            if not os.path.exists(self.save_filename):
                raise FileNotFoundError("Train-test split file not found.")
            with open(self.save_filename, 'r') as fp:
                data = json.load(fp)
            cv_fold_data = {
                'train_blocks': [int(i) for i in train_ids], 
                'val_blocks': [int(i) for i in val_ids]
            }
            data.setdefault('cv_folds', []).append(cv_fold_data)
            with open(self.save_filename, 'w') as fp:
                json.dump(data, fp, indent=2)
            logger.info(f"Appended CV fold to {self.save_filename}")

    def get_train_test_split(self) -> None:
        """
        Performs the primary train/test split based on the unified strata.
        From each stratum, 1 block is assigned to the test set.
        """
        test_block_ids = []

        for stratum_label, blocks_in_stratum in self.strata.items():
            if len(blocks_in_stratum) < self.stratum_size:
                logger.warning(
                    f"Stratum {stratum_label} is incomplete (size {len(blocks_in_stratum)}). "
                    "It will still be split, but results will be unbalanced."
                )
            
            # Randomly choose 1 block from this stratum for the final test set
            test_sample = self.train_test_rng.choice(blocks_in_stratum, size=1, replace=False)[0]
            test_block_ids.append(test_sample)

        self.test_block_ids = sorted(test_block_ids)
        self.train_block_ids = sorted(list(np.setdiff1d(self.block_ids, self.test_block_ids)))

        # Saving train-test split
        self._save_block_split('train_test', self.train_block_ids, [], self.test_block_ids)

        logger.info(
            f"Initial split performed. Successfully created, stored, and saved "
            f"{len(self.train_block_ids)} train blocks."
            f"and {len(self.test_block_ids)} test blocks.")
        
        return None
    
    def get_cv_splits(self) -> None:
        """
        Creates all CV folds at once and stores them in `self.CV_splits`.
        This method is randomized and handles incomplete strata correctly.

        For each "full" stratum (with enough blocks for 1 validation block per fold),
        it randomly assigns one block to each fold's validation set.

        For "incomplete" strata (e.g., the last stratum if the total number of
        blocks is not a multiple of stratum_size), it randomly assigns each
        available block to one of the existing fold's validation sets.
        """
        if self.train_block_ids is None:
            raise RuntimeError("You must call the .get_train_test_split() method first.")
        
        # Initialize empty folds
        all_folds: List[CVFold] = [
            {"train_block_ids": [], "val_block_ids": []} for _ in range(self.n_splits)
        ]

        # Separate strata into those that are "full" enough for CV and those that are not.
        full_cv_strata = {}
        incomplete_cv_blocks = []
        for stratum_label, blocks_in_stratum in self.strata.items():
            # Find blocks from this stratum that are available for cross-validation
            cv_blocks = list(np.intersect1d(blocks_in_stratum, self.train_block_ids))
            
            # If a stratum has one block per fold, it's a "full" CV stratum
            if len(cv_blocks) == self.n_splits:
                full_cv_strata[stratum_label] = cv_blocks
            else:
                incomplete_cv_blocks.extend(cv_blocks)

        # --- Process full strata ---
        # For each full stratum, shuffle the available blocks and assign one to each fold
        for stratum_label, cv_blocks in full_cv_strata.items():
            self.validation_rng.shuffle(cv_blocks) # Randomize the block order
            
            for k in range(self.n_splits):
                val_block = cv_blocks[k]
                # Add the selected block to the validation set of fold k
                all_folds[k]["val_block_ids"].append(val_block)
                
                # Add all other blocks from this stratum to the training set of fold k
                train_blocks = [b for b in cv_blocks if b != val_block]
                all_folds[k]["train_block_ids"].extend(train_blocks)

        # --- Process incomplete strata ---
        # For each leftover block, randomly assign it to one fold's validation set
        if incomplete_cv_blocks:
            logger.info(
                f"Handling {len(incomplete_cv_blocks)} blocks from incomplete strata."
            )
            for block in incomplete_cv_blocks:
                # Randomly pick a fold for this block to be in validation
                val_fold_idx = self.validation_rng.choice(self.n_splits)
                
                # Assign this block to the validation set of the chosen fold
                all_folds[val_fold_idx]["val_block_ids"].append(block)
                
                # Assign this block to the training set of all other folds
                for k in range(self.n_splits):
                    if k != val_fold_idx:
                        all_folds[k]["train_block_ids"].append(block)
        
        # Finalize by sorting the block IDs in each fold for consistency
        for fold in all_folds:
            fold["train_block_ids"].sort()
            fold["val_block_ids"].sort()
            
        self.CV_splits = all_folds
        
        # Saving each fold
        for fold in all_folds:
            self._save_block_split('cv', fold['train_block_ids'], fold['val_block_ids'], [])

        logger.info(f"Successfully created, stored, and saved {len(self.CV_splits)} randomized CV folds.")

        return None
    
    def get_single_split(self) -> Tuple[List[int], List[int]]:
        """
        Creates a single, stratified train/validation split from the main training set.

        This is a convenience method for when a full cross-validation is not needed.
        It samples one block from each stratum (that is present in the main training set)
        to form the validation set. It uses the dedicated cross-validation RNG.

        You must call .get_train_test_split() before this method.

        Returns:
            A tuple containing (train_block_ids, val_block_ids).
        """
        if self.train_block_ids is None:
            raise RuntimeError("You must call the .get_train_test_split() method first.")

        val_block_ids = []
        for stratum_label, blocks_in_stratum in self.strata.items():
            # Find the blocks from this stratum that are part of the main training set.
            available_blocks = np.intersect1d(blocks_in_stratum, self.train_block_ids)

            # If there are any available blocks, pick one for the validation set.
            if available_blocks.size == 0:
                logger.warning(f"No available blocks for stratum {stratum_label}")
                continue
            else:
                if available_blocks.size == 1:
                    probability_of_inclusion = 1 / (self.stratum_size - 1)
                    if self.validation_rng.random() < probability_of_inclusion:
                        val_block_ids.append(available_blocks[0])
                else:
                    val_sample = self.validation_rng.choice(available_blocks, size=1, replace=False)[0]
                    val_block_ids.append(val_sample)

        # The final training set is the main training set minus the new validation set.
        final_train_ids = sorted(list(np.setdiff1d(self.train_block_ids, val_block_ids)))
        final_val_ids = sorted(val_block_ids)

        logger.info(
            f"Created single split: {len(final_train_ids)} train blocks, "
            f"{len(final_val_ids)} validation blocks."
        )
        return final_train_ids, final_val_ids
    
    def split(self) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Convenience method: ensures splits are created, then yields
        (train_block_ids, val_block_ids) for each CV fold.
        """
        if self.train_block_ids is None:
            self.get_train_test_split()
        if self.CV_splits is None:
            self.get_cv_splits()
            
        for fold in self.CV_splits:
            yield fold["train_block_ids"], fold["val_block_ids"]
    
    def _get_indices_from_blocks(self, block_ids: List[int]) -> List[int]:
            """
            Helper method to retrieve all bucket_indices from a list of block IDs.
            
            Args:
                block_ids: A list of block IDs to look up.
                
            Returns:
                A single, sorted list of all corresponding bucket_indices.
            """
            if not block_ids:
                return []
                
            all_indices = []
            for block_id in block_ids:
                all_indices.extend(self.blocks[block_id]['bucket_indices'])
            
            return sorted(all_indices)
    
    def get_test_idx(self) -> List[int]:
        """
        Returns the bucket indices for the single, final test set.
        
        This method must be called after get_train_test_split().
        """
        if self.test_block_ids is None:
            raise RuntimeError("You must call .get_train_test_split() first to create the test set.")
        
        logger.info(f"Retrieving bucket indices for {len(self.test_block_ids)} test blocks.")
        return self._get_indices_from_blocks(self.test_block_ids)

    def get_train_idx(self) -> Generator[List[int], None, None]:
        """
        Generator that yields the training bucket indices for each CV fold.
        
        You must call .get_cv_splits() before using this method.
        """
        if self.CV_splits is None:
            raise RuntimeError("You must call .get_cv_splits() first to create CV folds.")
        
        logger.info("Yielding train bucket indices for each CV fold.")
        for fold in self.CV_splits:
            yield self._get_indices_from_blocks(fold["train_block_ids"])

    def get_val_idx(self) -> Generator[List[int], None, None]:
        """
        Generator that yields the validation bucket indices for each CV fold.
        
        You must call .get_cv_splits() before using this method.
        """
        if self.CV_splits is None:
            raise RuntimeError("You must call .get_cv_splits() first to create CV folds.")
            
        logger.info("Yielding validation bucket indices for each CV fold.")
        for fold in self.CV_splits:
            yield self._get_indices_from_blocks(fold["val_block_ids"])