from rdkit import Chem
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np
import torch
import copy
import os
from tqdm import tqdm
import matplotlib

from src.evaluation.metrics import CondMolGenMetric
from src.callbacks.validation_callback import reconstruct_mol_and_filter_invalid_mr_bond
from src.utils import transforms as trans
from src.utils.reconstruct import reconstruct_from_generated_with_bond

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
        self.gen_idx = 0
    
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
        # self.outputs.extend(outputs)
        
        results, fail_results, recon_dict = reconstruct_mol_and_filter_invalid_mr_bond(outputs)

        os.makedirs(self.output_path, exist_ok=True)
        
        rank = trainer.global_rank

        for res in tqdm(results, desc="Batch Chem eval"):
            try:
                mol = res['mol']
                ligand_filename = res['ligand_filename']
                mol.SetProp('_Name', ligand_filename)
            
                Chem.SanitizeMol(mol)
                Chem.AssignStereochemistryFrom3D(mol)
                smiles = Chem.MolToSmiles(mol)
                validity = smiles is not None
                complete = '.' not in smiles
            except:
                print('sanitize failed')
                continue

            if not validity or not complete:
                print('validity', validity, 'complete', complete)
                continue

            out_fn = os.path.join(self.output_path, f'{rank}_{self.gen_idx}.sdf')
            with Chem.SDWriter(out_fn) as w:
                w.write(mol)
            self.gen_idx += 1

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_start(trainer, pl_module)
        self.outputs = []
        self.gen_idx = 0

    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        pass
        # results, fail_results, recon_dict = reconstruct_mol_and_filter_invalid_mr_bond(self.outputs)

        # # path = pl_module.cfg.accounting.test_outputs_dir
        # # if os.path.exists(path):
        # #     shutil.rmtree(path)
        # # os.makedirs(path, exist_ok=True)

        # # if os.path.exists(OUT_DIR):
        # #     shutil.rmtree(OUT_DIR)
        # os.makedirs(self.output_path, exist_ok=True)
        
        # rank = trainer.global_rank

        # for idx, res in enumerate(tqdm(results, desc="Chem eval")):
            
        #     try:
        #         mol = res['mol']
        #         ligand_filename = res['ligand_filename']
        #         mol.SetProp('_Name', ligand_filename)
            
        #         Chem.SanitizeMol(mol)
        #         Chem.AssignStereochemistryFrom3D(mol)
        #         smiles = Chem.MolToSmiles(mol)
        #         validity = smiles is not None
        #         complete = '.' not in smiles
        #     except:
        #         print('sanitize failed')
        #         continue

        #     if not validity or not complete:
        #         print('validity', validity, 'complete', complete)
        #         continue

        #     out_fn = os.path.join(self.output_path, f'{rank}_{idx}.sdf')
        #     with Chem.SDWriter(out_fn) as w:
                # w.write(mol)


class TrajectoryCallback(Callback):
    def __init__(self, output_path):
        super().__init__()
        self.output_path = os.path.join(output_path, 'traj')
        self.gen_idx = 0

    def on_test_start(self, trainer, pl_module):
        self.gen_idx = 0
        pl_module.trajectory_callback = self.save_trajectory

    def on_test_end(self, trainer, pl_module):
        pl_module.trajectory_callback = None

    def save_trajectory(self, pl_module, batch, sample_chain, offset):
        out_batch = copy.deepcopy(batch).cpu()
        ligand = out_batch['ligand']
        rgroup_mask = ligand.rgroup_mask
        scaf_fc_bond_mask = ligand.scaf_fc_bond_mask
        offset = offset.detach().cpu()[ligand.batch[rgroup_mask]]
        normalizer = torch.tensor(pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32)
        ligand.pos = ligand.pos * normalizer

        paths = []
        for idx in range(out_batch.num_graphs):
            path = os.path.join(self.output_path, f'{pl_module.global_rank}_{self.gen_idx + idx}')
            os.makedirs(path, exist_ok=True)
            paths.append(path)

        # The last entry is the extra final prediction, after all sampling steps.
        for step, (pos, atom_type, bond_type) in enumerate(sample_chain[:-1], start=1):
            ligand.pos[rgroup_mask] = (pos.detach().cpu() + offset) * normalizer
            ligand.x[rgroup_mask] = atom_type.argmax(dim=-1).detach().cpu()
            ligand.fc_bond_types[~scaf_fc_bond_mask] = bond_type.argmax(dim=-1).detach().cpu()

            for idx, path in enumerate(paths):
                start, end = ligand.ptr[idx:idx + 2].tolist()
                bond_mask = ligand.fc_bond_types_batch == idx
                mol = reconstruct_from_generated_with_bond(
                    ligand.pos[start:end].numpy().astype('float64'),
                    trans.get_atomic_number_from_index(
                        ligand.x[start:end], mode=pl_module.cfg.data.transform.ligand_atom_mode
                    ),
                    (out_batch['ligand', 'ligand'].edge_index[:, bond_mask] - start).numpy(),
                    ligand.fc_bond_types[bond_mask].numpy(),
                )
                mol.SetProp('_Name', out_batch.uuid[idx])
                with Chem.SDWriter(os.path.join(path, f'step_{step:03d}.sdf')) as writer:
                    # Keep intermediate frames even when their valence is invalid.
                    writer.SetKekulize(False)
                    writer.write(mol)

        self.gen_idx += out_batch.num_graphs
