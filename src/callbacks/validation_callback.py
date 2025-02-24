from rdkit import Chem
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from tqdm import tqdm
import numpy as np
import torch
import os
import pickle as pkl
import json
import matplotlib
import wandb
import copy
import glob
import shutil
import time

from src.evaluation.metrics import CondMolGenMetric
from src.evaluation.utils import convert_atomcloud_to_mol_smiles, save_mol_list
from src.evaluation.visualization import visualize, visualize_chain
from src.utils import transforms as trans 
from src.evaluation.utils import timing
from src.utils.reconstruct import reconstruct_from_generated, reconstruct_from_generated_with_bond, MolReconError

# this file contains the model which we used to visualize the

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# TODO: refactor and move center_pos (and that in train_bfn.py) into utils
def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, rgroup_mask, mode="protein"):
    # TODO
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    elif mode == "scaffold":
        offset = scatter_mean(ligand_pos[~rgroup_mask], batch_ligand[~rgroup_mask], dim=0)
        if batch_protein.numel() > 0:
            protein_pos = protein_pos - offset[batch_protein]
        else:
            protein_pos = protein_pos
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


def reconstruct_mol_and_filter_invalid_mr_bond(out_list):
    results = []
    n_recon, n_complete, n_valid = 0, 0, 0
    n_total = len(out_list)
    center_change_list, mol_pos_range_list = [], []

    for item in out_list:
        ligand_filename = item.uuid
        pos, atom_type, bond_type = (
            item["ligand"].pos.detach().cpu().numpy().astype('float64'),
            item["ligand"].atom_type.detach().cpu().numpy().astype('int32'), # [N]
            item["ligand"].fc_bond_types.detach().cpu().numpy().astype('int32'), # [N, N]
        )
        
        bond_index = item["ligand", "ligand"].edge_index.detach().cpu().numpy().astype('int32')
        
        if "protein" in item.node_types:
            protein_pos, protein_v = (
                item["protein"].pos.detach().cpu().numpy().astype('float64'), 
                item["protein"].x.detach().cpu().numpy(),
            )
        else:
            protein_pos = np.empty([0, 3], dtype=pos.dtype)
            protein_v = np.empty([0, 27], dtype=atom_type.dtype)
        
        # # remove fake atoms
        # pos = pos[atom_type != 0]
        # is_aromatic = is_aromatic[atom_type != 0]
        # atom_type = atom_type[atom_type != 0]
        
        try:
            mol = reconstruct_from_generated_with_bond(pos, atom_type, bond_index, bond_type)
            n_recon += 1

            mol_center = pos.mean(axis=0)
            if protein_pos.size == 0:
                protein_center = pos.mean(axis=0)
            else:
                protein_center = protein_pos.mean(axis=0)
            center_change = np.linalg.norm(mol_center - protein_center)
            mol_pos_range = np.linalg.norm(pos.max(axis=0)[0] - pos.min(axis=0)[0])

            res = {
                'mol': mol, 'ligand_filename': ligand_filename, 
                'pred_pos': pos, 'pred_v': atom_type, 'pred_bond': bond_type, 'bond_index': bond_index,
                'protein_center': protein_center, 'mol_center': mol_center,
                'center_change': center_change, 'mol_pos_range': mol_pos_range,
            }
            center_change_list.append(center_change)
            mol_pos_range_list.append(mol_pos_range)

            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistryFrom3D(mol)
            smiles = Chem.MolToSmiles(mol)
            complete = smiles is not None and '.' not in smiles
            validity = smiles is not None

            n_complete += int(complete)
            n_valid += int(validity)
            res['smiles'] = smiles                    
            res['complete'] = complete
            res['validity'] = validity
            results.append(res)
        except Exception as e:
            print(e)
            continue

    return results, {
        'recon_success': n_recon / n_total,
        'completeness': n_complete / n_total,
        'validity': n_valid / n_total,
        'center_change': np.mean(center_change_list),
        'mol_pos_range': np.mean(mol_pos_range_list),
    }
    

def reconstruct_mol_and_filter_invalid_mr(out_list):
    results = []
    n_recon, n_complete, n_valid = 0, 0, 0
    n_total = len(out_list)
    center_change_list, mol_pos_range_list = [], []

    for item in out_list:
        ligand_filename = item.uuid
        pos, atom_type, is_aromatic = (
            item["ligand"].pos.detach().cpu().numpy().astype('float64'),
            item["ligand"].atom_type.detach().cpu().numpy().astype('int32'), # [N]
            item["ligand"].is_aromatic.detach().cpu().numpy().astype('bool'), # [N]
        )
        
        if "protein" in item.node_types:
            protein_pos, protein_v = (
                item["protein"].pos.detach().cpu().numpy().astype('float64'), 
                item["protein"].x.detach().cpu().numpy(),
            )
        else:
            protein_pos = np.empty([0, 3], dtype=pos.dtype)
            protein_v = np.empty([0, 27], dtype=atom_type.dtype)
        
        # # remove fake atoms
        # pos = pos[atom_type != 0]
        # is_aromatic = is_aromatic[atom_type != 0]
        # atom_type = atom_type[atom_type != 0]

        try:
            mol = reconstruct_from_generated(pos, atom_type, is_aromatic, basic_mode=False)
            n_recon += 1

            mol_center = pos.mean(axis=0)
            if protein_pos.size == 0:
                protein_center = pos.mean(axis=0)
            else:
                protein_center = protein_pos.mean(axis=0)
            center_change = np.linalg.norm(mol_center - protein_center)
            mol_pos_range = np.linalg.norm(pos.max(axis=0)[0] - pos.min(axis=0)[0])

            res = {
                'mol': mol, 'ligand_filename': ligand_filename, 
                'pred_pos': pos, 'pred_v': atom_type, 'is_aromatic': is_aromatic,
                'protein_center': protein_center, 'mol_center': mol_center,
                'center_change': center_change, 'mol_pos_range': mol_pos_range,
            }
            center_change_list.append(center_change)
            mol_pos_range_list.append(mol_pos_range)

            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            complete = smiles is not None and '.' not in smiles
            validity = smiles is not None

            n_complete += int(complete)
            n_valid += int(validity)
            res['smiles'] = smiles                    
            res['complete'] = complete
            res['validity'] = validity
            results.append(res)
        except Exception as e:
            continue

    return results, {
        'recon_success': n_recon / n_total,
        'completeness': n_complete / n_total,
        'validity': n_valid / n_total,
        'center_change': np.mean(center_change_list),
        'mol_pos_range': np.mean(mol_pos_range_list),
    }

def reconstruct_mol_and_filter_invalid(out_list):
    results = []
    n_recon, n_complete, n_valid = 0, 0, 0
    n_total = len(out_list)
    center_change_list, mol_pos_range_list = [], []

    for item in out_list:
        ligand_filename, pos, atom_type, is_aromatic = item.ligand_filename, item.pos, item.atom_type, item.is_aromatic
        protein_pos, protein_v = item.protein_pos, item.protein_atom_feature
        
        pos = pos.cpu().numpy().astype('float64')
        atom_type = atom_type.cpu().numpy().astype('int32')
        is_aromatic = is_aromatic.cpu().numpy().astype('bool')
        protein_pos = protein_pos.cpu().numpy().astype('float64')
        # TODO turn off basic_mode = False to use predicted aromaticity
        try:
            mol = reconstruct_from_generated(pos, atom_type, is_aromatic, basic_mode=True)
            n_recon += 1

            mol_center = pos.mean(axis=0)
            protein_center = protein_pos.mean(axis=0)
            center_change = np.linalg.norm(mol_center - protein_center)
            mol_pos_range = np.linalg.norm(pos.max(axis=0)[0] - pos.min(axis=0)[0])

            res = {
                'mol': mol, 'ligand_filename': ligand_filename, 
                'pred_pos': pos, 'pred_v': atom_type, 'is_aromatic': is_aromatic,
                'protein_center': protein_center, 'mol_center': mol_center,
                'center_change': center_change, 'mol_pos_range': mol_pos_range,
            }
            center_change_list.append(center_change)
            mol_pos_range_list.append(mol_pos_range)

            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            complete = smiles is not None and '.' not in smiles
            validity = smiles is not None

            n_complete += int(complete)
            n_valid += int(validity)
            res['smiles'] = smiles                    
            res['complete'] = complete
            res['validity'] = validity
            results.append(res)
        except Exception as e:
            continue

    return results, {
        'recon_success': n_recon / n_total,
        'completeness': n_complete / n_total,
        'validity': n_valid / n_total,
        'center_change': np.mean(center_change_list),
        'mol_pos_range': np.mean(mol_pos_range_list),
    }


# TODO merge with ReconValidationCallback
class ValidationCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config, val_freq, bond_prediction) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = copy.deepcopy(docking_config)
        self.docking_config.mode = 'vina_score'
        self.val_freq = val_freq
        self.bond_prediction = bond_prediction
        self.outputs = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self.metric = CondMolGenMetric(
            atom_decoder=self.atom_decoder, # not used 
            atom_enc_mode=self.atom_enc_mode, # not used
            type_one_hot=self.type_one_hot, # False but not used
            single_bond=self.single_bond, # True for check stability
            docking_config=self.docking_config, # contan protein_root
        )

    def on_train_batch_start(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
        ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    @torch.no_grad()
    def calc_recon_loss(self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        
        with torch.no_grad():
            pl_module.dynamics.eval()
            sum_batches, sum_loss, sum_loss_pos, sum_loss_type, sum_loss_bond = 0, 0., 0., 0., 0.
            pos_normalizer = torch.tensor(
                pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=pl_module.device,
            )

            for batch in tqdm(trainer.val_dataloaders, desc='Recon_Val', leave=False):
                # prepare batch data
                batch = batch.to(pl_module.device)

                if "protein" in batch.node_types:
                    batch["protein"].pos = batch["protein"].pos / pos_normalizer
                batch["ligand"].pos = batch["ligand"].pos / pos_normalizer

                ligand_pos, ligand_v, batch_ligand, rgroup_mask, ligand_fc_bond, ligand_fc_bond_batch, scaf_fc_bond_mask = (
                    batch["ligand"].pos,
                    batch["ligand"].x,
                    batch["ligand"].batch,
                    batch["ligand"].rgroup_mask,
                    
                    batch["ligand"].fc_bond_types,
                    batch["ligand"].fc_bond_types_batch,
                    batch["ligand"].scaf_fc_bond_mask,
                )
                ligand_fc_edge_index = batch["ligand", "ligand"].edge_index
                            
                if "protein" in batch.node_types:
                    protein_pos, protein_v, batch_protein = (
                        batch["protein"].pos,
                        batch["protein"].x.float(),
                        batch["protein"].batch
                    )  
                
                else:
                    protein_pos = torch.empty([0, 3], dtype=ligand_pos.dtype, device=ligand_pos.device)
                    protein_v = torch.empty([0, 27], dtype=ligand_v.dtype, device=ligand_v.device).float()
                    batch_protein = torch.empty([0], dtype=ligand_pos.dtype, device=ligand_pos.device)
            
                # move protein center to origin & ligand correspondingly
                protein_pos, ligand_pos, offset = center_pos(
                    protein_pos, 
                    ligand_pos, 
                    batch_protein, 
                    batch_ligand,
                    rgroup_mask,
                    mode=pl_module.cfg.dynamics.center_pos_mode
                ) #TODO: ugly 
                
                num_graphs = batch_ligand.max().item() + 1
                sum_batches += num_graphs * (pl_module.cfg.dynamics.discrete_steps // 10)
                
                # go through the sampling time steps for loss computation instead of selecting random t
                for t in tqdm(range(0, pl_module.cfg.dynamics.discrete_steps, 10), desc='Recon'):
                    t = torch.tensor(
                        [t / float(pl_module.cfg.dynamics.discrete_steps)], 
                        dtype=ligand_pos.dtype, device=ligand_pos.device
                    ).repeat(num_graphs, 1)
                    
                    if not pl_module.cfg.dynamics.use_discrete_t and not pl_module.cfg.dynamics.destination_prediction:
                        t = torch.clamp(t, min=pl_module.dynamics.t_min)  # clamp t to [t_min,1]
                    
                    t_node = t.index_select(0, batch_ligand[rgroup_mask])
                    t_bond = t.index_select(0, ligand_fc_bond_batch[~scaf_fc_bond_mask])
                    # compute bfn loss  # TODO: convert to reconstruction loss
                    c_loss, d_loss_atom, d_loss_bond = pl_module.dynamics.reconstruction_loss_one_step(
                        t_node,
                        t_bond,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        rgroup_mask=rgroup_mask,
                        scaf_fc_bond_mask=scaf_fc_bond_mask,
                        ligand_fc_bond=ligand_fc_bond,
                        ligand_fc_bond_batch=ligand_fc_bond_batch,
                        ligand_fc_edge_index=ligand_fc_edge_index,
                    )
                    
                    loss = torch.mean(c_loss + pl_module.cfg.train.v_loss_weight * d_loss_atom + d_loss_bond)
                    sum_loss += float(loss) * num_graphs
                    sum_loss_pos += float(c_loss.sum())
                    sum_loss_type += float(d_loss_atom.sum())
                    sum_loss_bond += float(d_loss_bond.sum())

            recon_loss = {
                "val/recon_loss": sum_loss / sum_batches,
                "val/recon_loss_pos": sum_loss_pos / sum_batches,
                "val/recon_loss_type": sum_loss_type / sum_batches,
                "val/recon_loss_bond": sum_loss_bond / sum_batches,
            }
            return recon_loss

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
        
        if trainer.global_step % self.val_freq == 0: 
            # perform a full validation
            recon_loss = self.calc_recon_loss(trainer, pl_module)
            pl_module.dynamics.train()

            pl_module.log_dict(
                recon_loss, 
                on_step=True,
                prog_bar=False, 
                batch_size=pl_module.cfg.train.batch_size,
            )
            print(json.dumps(recon_loss, indent=4))

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)  # num_samples * ([num_atoms_i, 3], [num_atoms_i, num_atom_types])

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        recon_loss = self.calc_recon_loss(trainer, pl_module)
        pl_module.log_dict(recon_loss, sync_dist=True)
        print(json.dumps(recon_loss, indent=4))

        if self.bond_prediction:
            results, recon_dict = reconstruct_mol_and_filter_invalid_mr_bond(self.outputs)
        else:
            results, recon_dict = reconstruct_mol_and_filter_invalid_mr(self.outputs)

        if len(results) == 0:
            print('skip validation, no mols are valid & complete')
            return

        epoch = pl_module.current_epoch
        path = os.path.join(pl_module.cfg.accounting.val_outputs_dir, f'epoch_{epoch}')
        # clear previous outputs if exists
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        torch.save(results, os.path.join(path, f'generated.pt'))
        
        out_metrics = self.metric.evaluate(results)
        torch.save(results, os.path.join(path, f'vina_docked.pt'))
        out_metrics.update(recon_dict)
        out_metrics = {f'val/{k}': v for k, v in out_metrics.items()}
        pl_module.log_dict(out_metrics, sync_dist=True)
        print(json.dumps(out_metrics, indent=4))
        json.dump(out_metrics, open(os.path.join(path, 'metrics.json'), 'w'), indent=4)


class VisualizeMolAndTrajCallback(Callback):
    # here the call back, we save the molecules and also draw the figures also to the wandb.
    def __init__(self, atom_decoder, colors_dic, radius_dic, type_one_hot=False) -> None:
        super().__init__()
        self.outputs = []
        self.named_chain_outputs = {}
        self.atom_decoder = atom_decoder
        self.colors_dic = colors_dic
        self.radius_dic = radius_dic
        self.type_one_hot = type_one_hot

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)
        pl_module.eval()
        if len(self.named_chain_outputs['y']) == 0 and pl_module.cfg.visual.visual_chain:
            # normalize the position
            pos_normalizer = torch.tensor(
                pl_module.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=batch.protein_pos.device
            )
            batch.protein_pos = batch.protein_pos / pos_normalizer
            batch.ligand_pos = batch.ligand_pos / pos_normalizer

            # prepare batch data
            protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
                batch.protein_pos, 
                batch.protein_atom_feature.float(), 
                batch.protein_element_batch, 
                batch.ligand_pos,
                batch.ligand_atom_feature_full, 
                batch.ligand_element_batch
            )

            # move protein center to origin & ligand correspondingly
            protein_pos, ligand_pos, offset = center_pos(
                protein_pos, ligand_pos, batch_protein, batch_ligand, mode=pl_module.cfg.dynamics.center_pos_mode) #TODO: ugly 
            num_graphs = batch_protein.max().item() + 1
    
            theta_chain, sample_chain, y_chain = pl_module.dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                n_nodes=num_graphs,
                ligand_pos=ligand_pos, # for debug only
                sample_steps=pl_module.cfg.evaluation.sample_steps,
                desc='MolVis',
            )

            # restore the protein position
            batch.protein_pos = batch.protein_pos * pos_normalizer

            for chain, chain_name in zip([theta_chain, sample_chain, y_chain], ['theta', 'sample', 'y']):
                for i in range(len(chain)):
                    pred_pos = chain[i][0]
                    one_hot = chain[i][1]
                    out_batch = copy.deepcopy(batch)
                    # restore the ligand position (in chain)
                    pred_pos = pred_pos * pos_normalizer

                    atom_type = one_hot.argmax(dim=-1)
                    # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
                    atom_type = trans.get_atomic_number_from_index(atom_type, mode=pl_module.cfg.data.transform.ligand_atom_mode)
                    atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in atom_type]
                    atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)
                    out_batch.x, out_batch.pos = atom_type, pred_pos
                    _slice_dict = {
                        "x": out_batch._slice_dict["ligand_element"],
                        "pos": out_batch._slice_dict["ligand_pos"],
                    }
                    _inc_dict = {"x": out_batch._inc_dict["ligand_element"], "pos": out_batch._inc_dict["ligand_pos"]}
                    out_batch._inc_dict.update(_inc_dict)
                    out_batch._slice_dict.update(_slice_dict)
                    
                    out_data_list = out_batch.to_data_list()
                    self.named_chain_outputs[chain_name].append(
                        out_data_list[0]
                    )  # always append the first sampled dtat

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []
        self.named_chain_outputs = {"theta": [], "sample": [], "y": []}

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        with timing('saving mol chain'):
            epoch = pl_module.current_epoch

            # save mols
            if pl_module.cfg.visual.save_mols:
                path = os.path.join(pl_module.cfg.accounting.generated_mol_dir, str(epoch))
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                # we save the figures here.
                save_mol_list(
                    path=path,
                    molecule_list=self.outputs,
                    index2atom=self.atom_decoder,
                    type_one_hot=self.type_one_hot,
                )
                if pl_module.cfg.visual.visual_nums > 0:
                    images = visualize(
                        path=path,
                        atom_decoder=self.atom_decoder,
                        color_dic=self.colors_dic,
                        radius_dic=self.radius_dic,
                        max_num=pl_module.cfg.visual.visual_nums,
                    )
                    # table = [[],[]]
                    table = []
                    for p_ in images:
                        im = plt.imread(p_)
                        table.append(wandb.Image(im))
                        # if len(table[0]) < 5:
                        #     table[0].append(wandb.Image(im))
                        # else:
                        #     table[1].append(wandb.Image(im))
                    # pl_module.logger.log_table(key="epoch {}".format(epoch), data=table, columns= ['1','2','3','4','5'])
                    pl_module.logger.log_image(key="epoch_{}".format(epoch), images=table)
                    # wandb.log()
                    # update to wandb
            
            # save chains
            if pl_module.cfg.visual.visual_chain:
                # we save the chains and visual the gif here.
                columns = list(self.named_chain_outputs.keys())
                chain_gifs = []

                # table = wandb.Table(columns=columns)
                for chain_name in columns:     
                    chain_path = os.path.join(
                        pl_module.cfg.accounting.generated_mol_dir, str(epoch), f"{chain_name}_chain"
                    )

                    if not os.path.exists(chain_path):
                        os.makedirs(chain_path, exist_ok=True)

                    save_mol_list(
                        path=chain_path,
                        molecule_list=self.named_chain_outputs[chain_name],
                        index2atom=self.atom_decoder,
                        type_one_hot=self.type_one_hot,
                    )
                    # if pl_module.cfg.visual.visual_nums > 0:
                    gif_path = visualize_chain(
                        path=chain_path,
                        atom_decoder=self.atom_decoder,
                        color_dic=self.colors_dic,
                        radius_dic=self.radius_dic,
                        spheres_3d=False,
                    )
                    gifs = wandb.Video(gif_path)
                    chain_gifs.append(gifs)
                
                pl_module.logger.log_table(
                    key="epoch_{}".format(epoch), data=[chain_gifs], columns=columns
                )

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)


class DockingTestCallback(Callback):
    def __init__(self, dataset, atom_enc_mode, atom_decoder, atom_type_one_hot, single_bond, docking_config) -> None:
        super().__init__()
        self.dataset = dataset
        self.atom_enc_mode = atom_enc_mode
        self.atom_decoder = atom_decoder
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.docking_config = docking_config
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

        results, recon_dict = reconstruct_mol_and_filter_invalid_mr(self.outputs)

        if len(results) == 0:
            print('skip validation, no mols are valid & complete')
            return

        path = pl_module.cfg.accounting.test_outputs_dir
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(path, timestr)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # dump config
        pl_module.cfg.save2yaml(os.path.join(path, 'config.yaml'))
        torch.save(results, os.path.join(path, f'generated.pt'))

        bad_case_dir = os.path.join(path, 'bad_cases')
        os.makedirs(bad_case_dir, exist_ok=True)
        print(f'bad cases dumped to {bad_case_dir}')

        out_metrics = self.metric.evaluate(results, bad_case_dir)
        torch.save(results, os.path.join(path, f'vina_docked.pt'))
        out_metrics.update(recon_dict)
        out_metrics = {f'test/{k}': v for k, v in out_metrics.items()}
        pl_module.log_dict(out_metrics, sync_dist=True)

        out_metrics['ckpt_path'] = pl_module.cfg.evaluation.ckpt_path
        out_metrics['test_outputs_dir'] = path
        out_metrics['sample_num_atoms'] = pl_module.cfg.evaluation.sample_num_atoms
        print(json.dumps(out_metrics, indent=4))
        json.dump(out_metrics, open(os.path.join(path, 'metrics.json'), 'w'), indent=4)

