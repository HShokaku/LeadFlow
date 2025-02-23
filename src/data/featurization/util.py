from typing import List, Tuple
from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem.rdchem import BondType
import numpy as np
import torch

# # Atom idx for one-hot encoding '#'-fake atom
# ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, 'B': 9, '#': 10}
# IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P', 9: 'B', 10: '#'}
# ATOM_NAME = ['C', 'O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'B', '#']

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

BOND_TYPES_TO_INDEX = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
}

INDEX_TO_BOND_TYPES = {v: k for k, v in BOND_TYPES_TO_INDEX.items()}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    ("Si", False): 0,
    ("C", False): 1,
    ("C", True): 2,
    ("O", False): 3,
    ("O", True): 4,
    ("N", False): 5,
    ("N", True): 6,
    ("F", False): 7,
    ("S", False): 8,
    ("S", True): 9,
    ("Cl", False): 10,
    ("Br", False): 11,
    ("I", False): 12,
    ("P", False): 13,
    ("P", True): 14,
    ("B", False): 15,
}

# 预测的时候先用index返回，再把原子类型和是否芳香还原
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}

# Atomic numbers
# CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15, 'B': 5, '#': 0}
# CHARGES_LIST = [6, 8, 7, 9, 16, 17, 35, 53, 15, 5, 0]

# def get_ligand_scaffold_mask(ligand: rdkit.Chem.Mol) -> torch.Tensor:
#     murcko_scaffold = MurckoScaffold.GetScaffoldForMol(ligand)
#     scaffold_substructures = ligand.GetSubstructMatches(murcko_scaffold)
#     substructure_indexes = [
#         i for substructure in scaffold_substructures for i in substructure
#     ]
#     mask = torch.zeros(len(ligand.GetAtoms()), dtype=torch.bool)
#     mask[substructure_indexes] = True
#     return mask


# def one_hot(x: List["str"], classes: List["str"]) -> torch.Tensor:
#     return torch.nn.functional.one_hot(
#         torch.tensor([classes.index(s) for s in x], dtype=torch.long), len(classes)
#     )

# def atomic_symbols_to_one_hot(symbols: List[str]) -> torch.FloatTensor:
#     return one_hot(symbols, ATOM_NAME)

def atomic_symbols_isAromatic_to_one_hot(symbols: List[Tuple[str, bool]]) -> torch.FloatTensor:
    return torch.nn.functional.one_hot(
        torch.tensor([MAP_ATOM_TYPE_AROMATIC_TO_INDEX[s] for s in symbols], dtype=torch.long), len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
    )

def atomic_symbols_isAromatic_to_index(symbols: List[Tuple[str, bool]]) -> torch.FloatTensor:
    return torch.tensor([MAP_ATOM_TYPE_AROMATIC_TO_INDEX[s] for s in symbols], dtype=torch.long)


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def transfer_conformers(scaf, mol):
    matches = mol.GetSubstructMatches(scaf)

    if len(matches) < 1:
        raise Exception('Could not find scaffold matches')
    
    match = matches[0]
    # for match in matches:
    mol_coords = mol.GetConformer().GetPositions()
    scaf_coords = mol_coords[np.array(match)]
    scaf_conformer = create_conformer(scaf_coords)

    return scaf_conformer


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):
        element = data["element"].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = torch.nn.functional.one_hot(data["atom_to_aa_type"], num_classes=self.max_num_aa)
        is_backbone = data["is_backbone"].view(-1, 1).long()
        # TODO: is_backbone is 0/1 values, not sure the feature is treated as categorical, if so, change to 2-hot
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        # data.protein_atom_feature = x
        
        return x