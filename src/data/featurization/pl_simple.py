import torch
import numpy as np
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import HeteroData

from src.data.featurization.util import (
    MAP_ATOM_TYPE_AROMATIC_TO_INDEX,
    atomic_symbols_isAromatic_to_index,
    FeaturizeProteinAtom,
)


class ProteinLigandSimpleFeaturization:
    def __init__(self) -> None:
        self.ligand_features = len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        self.protein_featurizer = FeaturizeProteinAtom()
        
    @staticmethod
    def get_ligand_atom_types(ligand: Chem.Mol) -> torch.Tensor:
        return atomic_symbols_isAromatic_to_index(
            [(atom.GetSymbol(), atom.GetIsAromatic()) for atom in ligand.GetAtoms()]
        )
    
    def get_protein_atom_features(self, protein: HeteroData) -> torch.Tensor:
        return self.protein_featurizer(protein)

    @staticmethod
    def get_ligand_positions(ligand: Chem.Mol) -> Tensor:
        return torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)
    

    def __call__(self, mol: Chem.Mol, scaffold_mol: Chem.Mol, pocket: dict, uuid: int) -> HeteroData:

        x = self.get_ligand_atom_types(mol)
        pos = self.get_ligand_positions(mol)

        if scaffold_mol is not None and mol.HasSubstructMatch(scaffold_mol):
            matches = mol.GetSubstructMatches(scaffold_mol)
            match = matches[0]

            # 如果全匹配上了，那么就不需要mask了
            if len(match) == len(x):
                print(f"Found scaffold all matches for {uuid}")
                match = []
        else:
            print(f"Could not find scaffold matches for {uuid}")
            match = []
        
        match_tensor = torch.tensor(match, dtype=torch.long)
        rgroup_mask = torch.ones(len(x), dtype=torch.bool)
        rgroup_mask[match_tensor] = False
        
        return HeteroData(
            protein={
                "x": self.get_protein_atom_features(pocket),
                "pos": pocket["pos"],
            },
            ligand={
                "x": x,
                "pos": pos,
                "rgroup_mask": rgroup_mask,
            },
            uuid=str(uuid)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atom_names={MAP_ATOM_TYPE_AROMATIC_TO_INDEX})"
    
    


