from torch_geometric.data import Dataset

from src.data.dataset.selfGenBench.dataset import selfGenBench_Dataset
from src.data.dataset.datamodule import ProteinLigandComplexDataModule


class selfGenBench_DataModule(ProteinLigandComplexDataModule):
    def __init__(
        self,
        root: str,
        pre_transform=None,
        pre_filter=None,
        batch_size=32,
        pocket_cutoff=10.,
        test_batch_size=None,
        val_batch_size=None,
        shuffle=True,
        overfit_item=False,
        debug=False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            val_batch_size=val_batch_size,
            shuffle=shuffle,
            overfit_item=overfit_item,
        )
        self.root = root
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.debug = debug
        self.pocket_cutoff = pocket_cutoff
        self.save_hyperparameters()

    def dataset_from_split(self, split: str) -> Dataset:
        return selfGenBench_Dataset(
            self.root,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            split=split,
            pocket_cutoff=self.pocket_cutoff,
            debug=self.debug,
        )