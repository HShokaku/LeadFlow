import copy
import numpy as np
import torch

from time import time
# from torch.profiler import profile, record_function, ProfilerActivity

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.data import HeteroData

from src.config.config import Config
from src.models.bfn4sbdd import BFN4SBDDScoreModel

import src.evaluation.utils.atom_num as atom_num
import src.utils.transforms as trans

from src.utils.train import get_optimizer, get_scheduler


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


class SBDDTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False

    def forward(self, x):
        pass

    def training_step(self, batch: HeteroData, batch_idx):
        ligand_pos, ligand_v, batch_ligand, rgroup_mask = (
            batch["ligand"].pos,
            batch["ligand"].x,
            batch["ligand"].batch,
            batch["ligand"].rgroup_mask,
        )
                    
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

        # t2 = time()

        # NOT USED
        with torch.no_grad():
            # add noise to protein_pos
            protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std # 0
            gt_protein_pos = protein_pos + protein_noise
            # random rotation as data aug
            if self.cfg.train.random_rot:
                M = np.random.randn(3, 3)
                Q, __ = np.linalg.qr(M)
                Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                gt_protein_pos = gt_protein_pos @ Q
                ligand_pos = ligand_pos @ Q

        num_graphs = batch.num_graphs
        # !!!!!
        gt_protein_pos, ligand_pos, _ = center_pos(
            gt_protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            rgroup_mask,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly

        # gt_protein_pos = gt_protein_pos / self.cfg.data.normalizer

        # t3 = time()
        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        ).index_select(
            0, batch_ligand[rgroup_mask]
        )  # different t for different molecules.

        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            # t = torch.randint(0, 999, [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device).index_select(0, batch_ligand) #different t for different molecules.
            # t = t / 1000.0
            # else:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]

        t4 = time()
        c_loss, d_loss, discretised_loss = self.dynamics.loss_one_step(
            t,
            protein_pos=gt_protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            rgroup_mask=rgroup_mask,
        )

        # here the discretised_loss is close for current version.

        loss = torch.mean(c_loss + self.cfg.train.v_loss_weight * d_loss + discretised_loss)

        # t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'loss': loss.item(),
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
        )
        self.log_dict(
            {
                'loss_pos': c_loss.mean().item(), 
                'loss_type': d_loss.mean().item(),
                'loss_c_ratio': c_loss.mean().item() / loss.item(),
            },
            on_step=True,
            prog_bar=False,
            batch_size=self.cfg.train.batch_size,
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return None
        self.train_losses.append(loss.clone().detach().cpu())

        # t0 = time()

        # if self.log_time:
        #     self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
        #     print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
        #     print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
        #     print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
        #     print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
        #     print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
        #     print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
        #     print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')
        return loss

    def validation_step(self, batch, batch_idx):
        out_data_list = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc=f'Val')
        return out_data_list
    
    def test_step(self, batch, batch_idx):
        # TODO change order, samples of the same pocket should be together, reduce protein loading
        out_data_list = []
        n_samples = self.cfg.evaluation.num_samples
        # repeat n_samples times sampling
        for _ in range(n_samples):
            batch_output = self.shared_sampling_step(batch, batch_idx, sample_num_atoms=self.cfg.evaluation.sample_num_atoms, 
                                                     desc=f'Test-{_}/{n_samples}')
            # for idx, item in enumerate(batch_output):
            out_data_list.append(batch_output)
                
        out_data_list_reorder = []
        for i in range(len(out_data_list[0])):
            for j in range(len(out_data_list)):
                out_data_list_reorder.append(out_data_list[j][i])
        return out_data_list_reorder

    def shared_sampling_step(self, batch: HeteroData, batch_idx, sample_num_atoms, desc=''):
        # here we need to sample the molecules in the validation step

        ligand_pos, ligand_v, batch_ligand, rgroup_mask = (
            batch["ligand"].pos,
            batch["ligand"].x,
            batch["ligand"].batch,
            batch["ligand"].rgroup_mask,
        )
                    
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
            
        num_graphs = batch_ligand.max().item() + 1  # B
        
        n_nodes = batch_ligand.size(0)  # N_lig
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"

        # move protein center to origin & ligand correspondingly
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            rgroup_mask,
            mode=self.cfg.dynamics.center_pos_mode,
        )  # TODO: ugly
         
        # determine the number of atoms in the ligand
        # TODO :change prior code here
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            rgroup_mask, ligand_pos, ligand_v = [], [], []
            for data_id in range(len(batch)):
                data = batch[data_id]
                rgroup_mask_single = data["ligand"].rgroup_mask
                ligand_v_single = data["ligand"].x[rgroup_mask_single]
                ligand_pos_single = data["ligand"].pos[rgroup_mask_single]
                raw_num_atoms = len(ligand_pos_single)
                new_num_atoms = atom_num.sample_atom_num(raw_num_atoms).astype(int)
                if new_num_atoms > raw_num_atoms:
                    new_num_atoms = raw_num_atoms
                    ligand_num_atoms.append(new_num_atoms)
                    # 在lignad_v_single里随便复制一点，放在最后，要复制new_num_atoms - raw_num_atoms次
                    ligand_v_single = torch.cat([ligand_v_single, (new_num_atoms - raw_num_atoms) * torch.ones_like(ligand_v_single[:1])])
                    ligand_pos_single = torch.cat([ligand_pos_single, (new_num_atoms - raw_num_atoms) * torch.randn_like(ligand_pos_single[:1])])
                    rgroup_mask_single = torch.zeros(len(ligand_v_single), dtype=torch.bool, device=ligand_pos_single.device)
                    rgroup_mask_single[-new_num_atoms:] = True
                pocket_size = atom_num.get_space_size(data["protein"].pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                # TODO 这里处理一下lignad_pos , ligand_v, batch_ligand, rgroup_mask
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(ligand_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=ligand_pos.device)
        elif sample_num_atoms == 'ref':
            # batch_ligand = batch.ligand_element_batch # 这边的batch_ligand就是pyg里的batch['ligand'].batch
            # 这边要做的是把batch_ligand用来统计每个分子的原子数，然后后面好做slicer
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(ligand_pos.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])# 这个经过cunsum之后就是每个分子的起始位置
        
        # TODO reuse for visualization and test
        # forward pass to get the ligand sample
        theta_chain, sample_chain, y_chain = self.dynamics.sample(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            rgroup_mask=rgroup_mask,
            # anchor_mask=anchor_mask,
            # n_nodes=n_nodes,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_graphs,
            # ligand_pos=ligand_pos,  # for debug only
            desc=desc,
        )
        # TODO:check offset and ligand_pos
        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        # only change the predicted rgroup's position, because the original ligand pos in batch did not change CoM
        pred_pos, one_hot = final[0] + offset[batch_ligand[rgroup_mask]], final[1]
        # copy a new batch, and put the rgroup's pred_pos into it, along with normalizer
        out_batch = copy.deepcopy(batch)
        
        out_batch["ligand"].pos[rgroup_mask] = pred_pos
        out_batch["ligand"].pos = out_batch["ligand"].pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
        )

        if "protein" in out_batch.node_types:
            out_batch["protein"].pos = out_batch["protein"].pos * torch.tensor(
                self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=ligand_pos.device
            )

        pred_v = one_hot.argmax(dim=-1) # get the index of the max value
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        
        out_batch["ligand"].x[rgroup_mask] = pred_v
        
        # 这里的函数要改成我们16个原子的，index -> atomic number, 这个是后面用来生成分子的
        pred_atom_type = trans.get_atomic_number_from_index(
            out_batch["ligand"].x, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]

        # for visualization
        # 这边是molcraft的legacy，就是把atom type的原子序数转回对应的index
        # atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        # atom_type = torch.tensor(atom_type, dtype=torch.long, device=ligand_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            out_batch["ligand"].x, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))
        
        # add necessary dict to new pyg batch
        # out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch["ligand"].atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=ligand_pos.device)
        out_batch["ligand"].is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=ligand_pos.device)
        # # out_batch.mol = results

        _slice_dict = {
            # "x": ligand_cum_atoms,
            # "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "is_aromatic": ligand_cum_atoms,
            # "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            # "x": out_batch._inc_dict["ligand_element"], # [0] * B, TODO: figure out what this is 
            # "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand"]["x"], # element对应的原子序数
            "is_aromatic": out_batch._inc_dict["ligand"]["x"],
            # "mol": out_batch._inc_dict["ligand_filename"],
        }
        out_batch._slice_dict["ligand"].update(_slice_dict)
        out_batch._inc_dict["ligand"].update(_inc_dict)
        out_data_list = out_batch.to_data_list()
        return out_data_list

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss,
            batch_size=self.cfg.train.batch_size,
        )
        self.train_losses = []

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)

        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
        }
