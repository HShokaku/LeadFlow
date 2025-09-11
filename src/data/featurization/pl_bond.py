import torch
import numpy as np
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import HeteroData

from src.data.featurization.util import (
    MAP_ATOM_TYPE_AROMATIC_TO_INDEX,
    BOND_TYPES_TO_INDEX,
    atomic_symbols_isAromatic_to_index,
    FeaturizeProteinAtom,
)


class ProteinLigandBondFeaturization:
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
    
    def get_ligand_fc_edge_index(self, ligand: Chem.Mol) -> Tensor:
        """
        Get the full connected edges index without self-loops for the ligand graph
        """
        edges = torch.combinations(torch.arange(ligand.GetNumAtoms()), r=2).T
        edges_rev = edges.flip(0)
        fc_edge_index = torch.cat([edges, edges_rev], dim=1)
        return fc_edge_index
    
    def get_ligand_fc_bond_types(self, ligand: Chem.Mol, fc_edge_index: Tensor) -> Tensor:
        row, col, bond_type = [], [], []
        for bond in ligand.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_type += 2 * [BOND_TYPES_TO_INDEX[bond.GetBondType()]]
            
        bond_index = torch.tensor([row, col], dtype=torch.long)
        bond_type = torch.tensor(bond_type, dtype=torch.long)
        
        # 0 = no bond, 1 = single bond, 2 = double bond, 3 = triple bond
        bond_matrix = torch.zeros((ligand.GetNumAtoms(), ligand.GetNumAtoms()), dtype=torch.long)
        bond_matrix[bond_index[0], bond_index[1]] = bond_type
        
        fc_bond_types = bond_matrix[fc_edge_index[0], fc_edge_index[1]]
        
        return fc_bond_types
    

    def __call__(self, mol: Chem.Mol, scaffold_mol: Chem.Mol, pocket: dict, uuid: int, mode='train') -> HeteroData:

        assert mode in ['train', 'val', 'test', 'de_novo', 'scaf_hopping'], f"Unknown featurize mode: {mode}"
        
        x = self.get_ligand_atom_types(mol)
        pos = self.get_ligand_positions(mol)

        if scaffold_mol is not None and mol.HasSubstructMatch(scaffold_mol):
            matches = mol.GetSubstructMatches(scaffold_mol)
            match = matches[0]


            if len(match) == len(x):
                # If the scaffold is the same as the ligand, we mask all atoms and bonds for denovo generation mode
                if mode == 'train' or mode == 'val' or mode == 'de_novo' or mode == 'scaf_hopping':
                    print(f"Found scaffold all matches for {uuid}")
                    match = []
                # In sample mode, if the scaffold is the same as the ligand, non atom or bond is masked, rgroup will be added in prior sampling mode
                elif mode == 'test':
                    match = range(len(x))
                else:
                    raise ValueError(f"Unknown mode for scaffold match: {mode}")
        else:
            print(f"Could not find scaffold matches for {uuid}")
            match = []
        
        # kekulize to transform aromatic bonds to single and double bonds
        Chem.Kekulize(mol)
        fc_edge_index = self.get_ligand_fc_edge_index(mol)
        fc_bond_types = self.get_ligand_fc_bond_types(mol, fc_edge_index)
        
        match_tensor = torch.tensor(match, dtype=torch.long)
        rgroup_mask = torch.ones(len(x), dtype=torch.bool)
        rgroup_mask[match_tensor] = False
        
        scaf_fc_bond_mask = (fc_edge_index[0][:, None] == match_tensor).any(1) & \
                            (fc_edge_index[1][:, None] == match_tensor).any(1)
        
        if mode == 'scaf_hopping':
            rgroup_mask = ~rgroup_mask
            scaf_fc_bond_mask = (~(fc_edge_index[0][:, None] == match_tensor).any(1)) & \
                                (~(fc_edge_index[1][:, None] == match_tensor).any(1))
        
        return HeteroData(
            protein={
                "x": self.get_protein_atom_features(pocket),
                "pos": pocket["pos"],
            },
            ligand={
                "x": x,
                "pos": pos,
                "rgroup_mask": rgroup_mask,
                # "fc_edge_index": fc_edge_index,
                "fc_bond_types": fc_bond_types,
                "scaf_fc_bond_mask": scaf_fc_bond_mask,
            },
            ligand__to__ligand={
                "edge_index": fc_edge_index,
                # "edge_attr": fc_bond_types,
            },
            uuid=str(uuid)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atom_names={MAP_ATOM_TYPE_AROMATIC_TO_INDEX})"
    
    


