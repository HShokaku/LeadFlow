from rdkit import Chem
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np
# import torch
import os
from tqdm import tqdm
import matplotlib

from src.evaluation.metrics import CondMolGenMetric
from src.callbacks.validation_callback import reconstruct_mol_and_filter_invalid_mr_bond

# this file contains the model which we used to visualize the

matplotlib.use("Agg")

import matplotlib.pyplot as plt



class DockingTestCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config, output_path, bond_prediction) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = docking_config
        self.output_path = output_path
        self.bond_prediction = bond_prediction
        self.outputs = []
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self.metric = CondMolGenMetric(
            atom_decoder=self.atom_decoder,
            atom_enc_mode=self.atom_enc_mode,
            type_one_hot=self.type_one_hot,
            single_bond=self.single_bond,
            docking_config=self.docking_config,
        )
    
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        self.outputs = []

    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)

        # TODO: !!! using openbabel to reconstruct the molecule
        results, fail_results, recon_dict = reconstruct_mol_and_filter_invalid_mr_bond(self.outputs)

        # path = pl_module.cfg.accounting.test_outputs_dir
        # if os.path.exists(path):
        #     shutil.rmtree(path)
        # os.makedirs(path, exist_ok=True)

        # if os.path.exists(OUT_DIR):
        #     shutil.rmtree(OUT_DIR)
        os.makedirs(self.output_path, exist_ok=True)
        
        rank = trainer.global_rank

        for idx, res in enumerate(tqdm(results, desc="Chem eval")):
            
            try:
                mol = res['mol']
                ligand_filename = res['ligand_filename']
                mol.SetProp('_Name', ligand_filename)
            
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                validity = smiles is not None
                complete = '.' not in smiles
            except:
                print('sanitize failed')
                continue

            if not validity or not complete:
                print('validity', validity, 'complete', complete)
                continue

            out_fn = os.path.join(self.output_path, f'{rank}_{idx}.sdf')
            with Chem.SDWriter(out_fn) as w:
                w.write(mol)

