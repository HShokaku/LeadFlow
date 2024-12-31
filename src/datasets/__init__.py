import torch
from torch.utils.data import Subset

from src.data.dataset import chembl_Dataset, chembl_DataModule
from src.data.featurization import ProteinLigandSimpleFeaturization
# from src.datasets.pl_pair_dataset import PocketLigandPairDataset, PocketLigandPairDatasetFeaturized, PocketLigandGeneratedPairDataset, PocketLigandPairDatasetFeaturized_mat2a
# from src.datasets.pdbbind import PDBBindDataset


# def get_dataset(config, *args, **kwargs):
#     name = config.name
#     root = config.path
#     ligand_atom_mode = config.transform.ligand_atom_mode
#     if name == 'pl':
#         dataset = PocketLigandPairDataset(root, *args, **kwargs)
#     elif name == 'pl_tr':
#         dataset = PocketLigandPairDatasetFeaturized(root, ligand_atom_mode=ligand_atom_mode,
#                                                      *args, **kwargs)
#         return dataset, {"train": dataset.train_data, "test": dataset.test_data}
#     elif name == 'mat2a':
#         dataset = PocketLigandPairDatasetFeaturized_mat2a(root, ligand_atom_mode=ligand_atom_mode,
#                                                      *args, **kwargs)
#         return dataset, {"train": dataset.train_data, "test": dataset.test_data}
#     elif name == 'pl_dcmp':
#         dataset = PocketLigandGeneratedPairDataset(root, *args, **kwargs)
#         return dataset, {"train": dataset, "test": dataset}
#     elif name == 'pdbbind':
#         dataset = PDBBindDataset(root, *args, **kwargs)
#     else:
#         raise NotImplementedError('Unknown dataset: %s' % name)
    
#     # print(config)

#     if config.with_split:
#         split = torch.load(config.split)
#         subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
#         return dataset, subsets
#     else:
#         return dataset

def get_datamodule(dataset_name: str, batch_size: int = 32):
    if "chembl" in dataset_name:
        dataset_constructor = chembl_DataModule
        pre_transform = ProteinLigandSimpleFeaturization()
        pre_filter = None
    # elif "crossdocked" in dataset_name:
    #     dataset_constructor = Multi_R_anchor_crossdock_DataModule
    #     pre_transform = ProteinLigandSimpleFeaturization()
    #     pre_filter = None
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")
    
    dataset = dataset_constructor(
        f"data/{dataset_name}/",
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
    )
    return dataset