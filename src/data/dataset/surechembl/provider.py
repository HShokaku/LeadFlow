import os
import pandas as pd
from pathlib import Path
from typing import Iterable, Iterator, List

from rdkit import Chem

from src.data.util import LMDBStorage


class MultiR_anchor_provider(Iterable):
    def __init__(
        self, table_path: Path, lmdb_path: Path
    ) -> None:
        super().__init__()
        self.table_path = table_path
        self.lmdb_path = lmdb_path
        
        self.lmdb = LMDBStorage(self.lmdb_path)
        self.table = pd.read_csv(self.table_path)
        
    def __iter__(self) -> Iterator[str]:
        return iter(self.table["uuid"])
    
    def get_keys(self) -> List[str]:
        return list(iter(self))
    
    def __len__(self) -> int:
        return len(self.table)
    

class GeomProvider(Iterable):
    def __init__(
        self, table_path: Path, frag_path: Path, linker_path: Path, mol_path: Path
    ) -> None:
        super().__init__()
        self.table_path = table_path
        self.frag_path = frag_path
        self.linker_path = linker_path
        self.mol_path = mol_path
        if not Path(self.table_path).exists():
            raise FileNotFoundError(f"Could not find {self.table_path}")
        if not Path(self.frag_path).exists():
            raise FileNotFoundError(f"Could not find {self.frag_path}")
        if not Path(self.linker_path).exists():
            raise FileNotFoundError(f"Could not find {self.linker_path}")
        if not Path(self.mol_path).exists():
            raise FileNotFoundError(f"Could not find {self.mol_path}")
        self.table = pd.read_csv(self.table_path)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self.table["uuid"])
    
    def get_keys(self) -> List[str]:
        return list(iter(self))

    def _read_sdf(self, sdf_path, sanitize=True, removeHs=True):
        if removeHs == True:
            assert sanitize == True, "sanitize must be True when removeHs is True"
        with Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize, removeHs=removeHs) as supplier:
            for molecule in supplier:
                yield molecule
    
    def get_generator(self):
        table = self.table
        frags = self._read_sdf(self.frag_path)
        linkers = self._read_sdf(self.linker_path)
        mol = self._read_sdf(self.mol_path, sanitize=True)
        generator = zip(table.iterrows(), frags, linkers, mol)
        
        return generator


