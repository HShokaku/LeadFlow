import os
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, List, Optional, Iterator

import torch
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import HeteroData

from src.utils.misc import torchify_dict
from src.data.util import LMDBStorage
from src.data.protein_ligand import PDBProtein
from src.data.dataset.dataset import ProteinLigandDataset
from src.data.featurization.multi_frag import prepare_scaffold_and_rgroup_list


# multi with anchor
class Multi_R_anchor_crossdock_Dataset(ProteinLigandDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        pocket_cutoff: float = 10.,
        log: bool = True,
        seed: int = 1,
        debug: bool = False,
    ):
        self.seed = seed
        self.debug = debug
        self.pocket_cutoff = pocket_cutoff
        self.randomised_train_test_val_split = None
        super().__init__(root, split, transform, pre_transform, pre_filter, log=log)

    
    def _load_splits(self):
        split_names = ["train", "val", "test"]
        
        splits = {}
        for split in split_names:
            splits[split] = {
                "table_path": Path(self.raw_dir) / f"{split}_debug.csv",
            }
        return splits
    
    def process(self):
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        self._featurize_split()
            
    
    def _get_provider(self, split: str) -> Iterator:
        if self.debug:
            table_path = Path(self.raw_dir) / f'{split}_debug.csv'
        else:
            table_path = Path(self.raw_dir) / f'{split}.csv'
            
        self._log(f"Reading table from {table_path}")
        table = pd.read_csv(table_path)
        num_rows = len(table)
        return table.iterrows(), num_rows
    
    def _featurize_split(self, split: str = None):
        if split is None:
            split = self.split
            
        processed_dir = Path(self.processed_dir)
        
        self.provider, num_candidates = self._get_provider(split)
        
        split_file = processed_dir / f"{split}.pt"

        if split_file.exists():
            self._log(f"Skipping {split} as {split_file} already exists")
            return

        self._log(
            f"Processing {num_candidates} complexes to {split_file} for {split}"
        )
        self._featurize_split_given_candidates(num_candidates, split_file)
        
    def _featurize_split_given_candidates(self, num_candidates: int, split_file: Path):
        transformed_data = []
        for _, row in (pbar := tqdm(self.provider, total=num_candidates)):
            candidate = self._featurize_candidate(row)
            if candidate is not None:
                transformed_data.append(candidate)
                pbar.set_description(
                    f"Transformed {len(transformed_data)} complexes"
                )
        transformed_data = [c for c in transformed_data if c is not None]
        self._log(f"Transformed {len(transformed_data)} complexes")
        data, slices = self.collate(transformed_data)
        torch.save((data, slices), split_file)
        
    def _featurize_candidate(self, row: pd.Series) -> Optional[HeteroData]:
        try:
            identifier = row['identifier']
            
            # TODO: 从identifier中读ligand.sdf和protein.pdb
            raw_dir = Path(self.raw_dir)
            ligand_fn = str(raw_dir / "crossdocked_pocket10" / f"{identifier}.sdf")
            protein_fn = str(raw_dir / "crossdocked_pocket10" / f"{identifier}_pocket10.pdb")
            
            mol = Chem.MolFromMolFile(ligand_fn)
            mol = Chem.RemoveAllHs(mol)
                        
            scaf_smi = row['scaffolds']
            dec_smi = row['decorations']
            # get the scaffold and rgroup mols here
            scaf_mol, rgroup_mol = prepare_scaffold_and_rgroup_list(scaf_smi, dec_smi, mol) # actually, dec_smi is not used in this function
            
            protein = PDBProtein(protein_fn)
            pdb_block_pocket = protein.residues_to_pdb_block(
                protein.query_residues_ligand(scaf_mol, radius=self.pocket_cutoff)
            )
            pocket = PDBProtein(pdb_block_pocket)
            pocket_dict = torchify_dict(pocket.to_dict_atom())
            
            
            if self.pre_transform is not None:
                data = self.pre_transform(scaf_mol, rgroup_mol, pocket_dict, identifier)
            
            return data
        except Exception as e:
            self._log(f"Could not process {identifier} in {self.split}: {e}")
            return None
        