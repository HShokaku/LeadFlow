import copy
import numpy as np
import torch
import torch.nn.functional as F

from time import time
# from torch.profiler import profile, record_function, ProfilerActivity

import pytorch_lightning as pl

from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.data import HeteroData, Batch

from src.config.config import Config
from src.models.bfn4sbdd_bond import BFN4SBDDScoreModel
from src.datasets.pl_data import FOLLOW_BATCH

import src.evaluation.utils.atom_num as atom_num
import src.utils.transforms as trans

from src.utils.train import get_optimizer, get_scheduler, skip_computation_on_oom


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


class SBDDTrainLoopBondDPO(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        self.ref_dynamics = BFN4SBDDScoreModel(**self.cfg.dynamics.todict())
        
        # [ time, h_t, pos_t, edge_index]
        self.train_losses = []
        self.save_hyperparameters(self.cfg.todict())
        self.time_records = np.zeros(6)
        self.log_time = False
    
        
    def setup(self, stage):
        if stage == "fit":
            if not self.cfg.train.resume:
                # using pretrained dynamics
                ckpt = torch.load(self.cfg.train.ft_ckpt_path, map_location="cpu")
                dynamics_state_dict = {k.replace("dynamics.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("dynamics.")}
                self.dynamics.load_state_dict(dynamics_state_dict)
                self.ref_dynamics.load_state_dict(self.dynamics.state_dict())
            
            self.ref_dynamics.eval()
            for param in self.ref_dynamics.parameters():
                param.requires_grad = False

            self.trainer.strategy.barrier()

    def forward(self, x):
        pass
    
    @skip_computation_on_oom(return_value=None, error_message="⚠️ OOM detected, skipping batch...")
    def training_step(self, batch: HeteroData, batch_idx):
        
        # winner
        ligand_pos_w, ligand_v_w, batch_ligand_w, rgroup_mask_w, ligand_fc_bond_w, ligand_fc_bond_batch_w, scaf_fc_bond_mask_w = (
            batch["w_ligand"].pos,
            batch["w_ligand"].x,
            batch["w_ligand"].batch,
            batch["w_ligand"].rgroup_mask,
            
            batch["w_ligand"].fc_bond_types,
            batch["w_ligand"].fc_bond_types_batch,
            batch["w_ligand"].scaf_fc_bond_mask,
        )
        ligand_fc_edge_index_w = batch["w_ligand", "w_ligand"].edge_index
                    
        protein_pos_w, protein_v_w, batch_protein_w = (
            batch["w_protein"].pos,
            batch["w_protein"].x.float(),
            batch["w_protein"].batch
        )
        
        # loser
        ligand_pos_l, ligand_v_l, batch_ligand_l, rgroup_mask_l, ligand_fc_bond_l, ligand_fc_bond_batch_l, scaf_fc_bond_mask_l = (
            batch["l_ligand"].pos,
            batch["l_ligand"].x,
            batch["l_ligand"].batch,
            batch["l_ligand"].rgroup_mask,
            
            batch["l_ligand"].fc_bond_types,
            batch["l_ligand"].fc_bond_types_batch,
            batch["l_ligand"].scaf_fc_bond_mask,
        )
        
        ligand_fc_edge_index_l = batch["l_ligand", "l_ligand"].edge_index
                    
        protein_pos_l, protein_v_l, batch_protein_l = (
            batch["l_protein"].pos,
            batch["l_protein"].x.float(),
            batch["l_protein"].batch
        )  

        # reward
        reward_w = batch["w_affinity"]
        reward_l = batch["l_affinity"]
        
        rewards = torch.stack([reward_w, reward_l], dim=1)
        rewards = F.softmax(rewards * self.cfg.train.beta_reward, dim=1)
        
        reward_w = rewards[:, 0]
        reward_l = rewards[:, 1]
        
        #########################################
        with torch.no_grad():
            # add noise to protein_pos
            protein_noise_w = torch.randn_like(protein_pos_w) * self.cfg.train.pos_noise_std # 0
            protein_noise_l = torch.randn_like(protein_pos_l) * self.cfg.train.pos_noise_std # 0
            gt_protein_pos_w = protein_pos_w + protein_noise_w
            gt_protein_pos_l = protein_pos_l + protein_noise_l
                
                
        num_graphs = batch.num_graphs

        # winner
        gt_protein_pos_w, ligand_pos_w, _ = center_pos(
            gt_protein_pos_w,
            ligand_pos_w,
            batch_protein_w,
            batch_ligand_w,
            rgroup_mask_w,
            mode=self.cfg.dynamics.center_pos_mode,
        )
        
        # loser
        gt_protein_pos_l, ligand_pos_l, _ = center_pos(
            gt_protein_pos_l,
            ligand_pos_l,
            batch_protein_l,
            batch_ligand_l,
            rgroup_mask_l,
            mode=self.cfg.dynamics.center_pos_mode,
        )

        t = torch.rand([num_graphs, 1], dtype=ligand_pos_w.dtype, device=ligand_pos_w.device)

        if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
            t = torch.clamp(t, min=self.dynamics.t_min)  # clamp t to [t_min,1]


        t_node_w = t.index_select(0, batch_ligand_w[rgroup_mask_w])
        t_bond_w = t.index_select(0, ligand_fc_bond_batch_w[~scaf_fc_bond_mask_w])
        
        t_node_l = t.index_select(0, batch_ligand_l[rgroup_mask_l])
        t_bond_l = t.index_select(0, ligand_fc_bond_batch_l[~scaf_fc_bond_mask_l])
        
        
        # winner_loss
        c_loss_w, d_loss_atom_w, d_loss_bond_w = self.dynamics.loss_one_step(
            t_node_w,
            t_bond_w,
            protein_pos=gt_protein_pos_w,
            protein_v=protein_v_w,
            batch_protein=batch_protein_w,
            ligand_pos=ligand_pos_w,
            ligand_v=ligand_v_w,
            batch_ligand=batch_ligand_w,
            rgroup_mask=rgroup_mask_w,
            scaf_fc_bond_mask=scaf_fc_bond_mask_w,
            ligand_fc_bond=ligand_fc_bond_w,
            ligand_fc_bond_batch=ligand_fc_bond_batch_w,
            ligand_fc_edge_index=ligand_fc_edge_index_w,
        )
        
        # loser_loss
        c_loss_l, d_loss_atom_l, d_loss_bond_l = self.dynamics.loss_one_step(
            t_node_l,
            t_bond_l,
            protein_pos=gt_protein_pos_l,
            protein_v=protein_v_l,
            batch_protein=batch_protein_l,
            ligand_pos=ligand_pos_l,
            ligand_v=ligand_v_l,
            batch_ligand=batch_ligand_l,
            rgroup_mask=rgroup_mask_l,
            scaf_fc_bond_mask=scaf_fc_bond_mask_l,
            ligand_fc_bond=ligand_fc_bond_l,
            ligand_fc_bond_batch=ligand_fc_bond_batch_l,
            ligand_fc_edge_index=ligand_fc_edge_index_l,
        )
        
        ############## Ref Loss ################
        
        # ref winner loss
        c_loss_ref_w, d_loss_atom_ref_w, d_loss_bond_ref_w = self.ref_dynamics.loss_one_step(
            t_node_w,
            t_bond_w,
            protein_pos=gt_protein_pos_w,
            protein_v=protein_v_w,
            batch_protein=batch_protein_w,
            ligand_pos=ligand_pos_w,
            ligand_v=ligand_v_w,
            batch_ligand=batch_ligand_w,
            rgroup_mask=rgroup_mask_w,
            scaf_fc_bond_mask=scaf_fc_bond_mask_w,
            ligand_fc_bond=ligand_fc_bond_w,
            ligand_fc_bond_batch=ligand_fc_bond_batch_w,
            ligand_fc_edge_index=ligand_fc_edge_index_w,
        )
        
        # ref loser loss
        c_loss_ref_l, d_loss_atom_ref_l, d_loss_bond_ref_l = self.ref_dynamics.loss_one_step(
            t_node_l,
            t_bond_l,
            protein_pos=gt_protein_pos_l,
            protein_v=protein_v_l,
            batch_protein=batch_protein_l,
            ligand_pos=ligand_pos_l,
            ligand_v=ligand_v_l,
            batch_ligand=batch_ligand_l,
            rgroup_mask=rgroup_mask_l,
            scaf_fc_bond_mask=scaf_fc_bond_mask_l,
            ligand_fc_bond=ligand_fc_bond_l,
            ligand_fc_bond_batch=ligand_fc_bond_batch_l,
            ligand_fc_edge_index=ligand_fc_edge_index_l,
        )
        # here the discretised_loss is close for current version.

        # TODO: check if the loss is (N, 1)
        # calculate the DPO loss
        c_loss = (c_loss_w - c_loss_ref_w) - (c_loss_l - c_loss_ref_l)
        d_loss_atom = (d_loss_atom_w - d_loss_atom_ref_w) - (d_loss_atom_l - d_loss_atom_ref_l)
        d_loss_bond = (d_loss_bond_w - d_loss_bond_ref_w) - (d_loss_bond_l - d_loss_bond_ref_l)
        
        c_loss = torch.mean(reward_w * (-F.logsigmoid(-1 * self.cfg.train.dpo_beta * c_loss)) +  \
            reward_l * (-F.logsigmoid(self.cfg.train.dpo_beta * c_loss)))
        d_loss_atom = torch.mean(reward_w * (-F.logsigmoid(-1 * self.cfg.train.dpo_beta * d_loss_atom)) +  \
            reward_l * (-F.logsigmoid(self.cfg.train.dpo_beta * d_loss_atom)))
        d_loss_bond = torch.mean(reward_w * (-F.logsigmoid(-1 * self.cfg.train.dpo_beta * d_loss_bond)) +  \
            reward_l * (-F.logsigmoid(self.cfg.train.dpo_beta * d_loss_bond)))
        

        loss = torch.mean(c_loss + self.cfg.train.v_loss_weight * d_loss_atom + d_loss_bond)

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
                'loss_atom': d_loss_atom.mean().item(),
                'loss_bond': d_loss_bond.mean().item(),
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
        batch_ligand = batch["ligand"].batch
                    
        if "protein" in batch.node_types:
            protein_pos, protein_v, batch_protein = (
                batch["protein"].pos,
                batch["protein"].x.float(),
                batch["protein"].batch
            )  
        
        else:
            # for no protein pretraining
            protein_pos = torch.empty([0, 3], dtype=ligand_pos.dtype, device=ligand_pos.device)
            protein_v = torch.empty([0, 27], dtype=ligand_v.dtype, device=ligand_v.device).float()
            batch_protein = torch.empty([0], dtype=ligand_pos.dtype, device=ligand_pos.device)
        
        # TODO: change to use batch_protein if denovo generation
        num_graphs = batch_ligand.max().item() + 1  # B
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"
         
        # determine the number of atoms in the ligand
        if sample_num_atoms == 'prior':
            ligand_num_atoms, ligand_num_edges = [], []
            rgroup_mask, ligand_pos, ligand_v = [], [], []
            scaf_fc_bond_mask, ligand_fc_bond = [], []
            data_batch = []
            while len(ligand_num_atoms) < len(batch):
                data_id = len(ligand_num_atoms)
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data["protein"].pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                num_atoms = atom_num.sample_atom_num(pocket_size).astype(int)
                num_scaf_atoms = len(data["ligand"].pos[~data["ligand"].rgroup_mask])
                update_num_atoms = num_atoms - num_scaf_atoms
                # update_num_atoms must > 0
                if update_num_atoms <= 0:
                    continue
                
                # ligand atom update
                ligand_num_atoms.append(num_atoms)
                # update ligand_pos and ligand_v, rgroup_mask
                data["ligand"].pos = torch.cat([data["ligand"].pos[~data["ligand"].rgroup_mask], torch.zeros(update_num_atoms, 3, device=data["ligand"].pos.device)])
                data["ligand"].x = torch.cat([data["ligand"].x[~data["ligand"].rgroup_mask], torch.zeros(update_num_atoms, device=data["ligand"].x.device)])
                # update rgroup_mask, (~data["ligand"].rgroup_mask).sum().item() is the number of atoms in the scaffold, so the rgroup_mask is the rest
                # data["ligand"].rgroup_mask = torch.arange(num_atoms, device=data["ligand"].rgroup_mask.device) >= (~data["ligand"].rgroup_mask).sum().item()
                
                # ligand edge index update
                scaf_indices = torch.where(data["ligand"].rgroup_mask == False)[0]
                indices_mapping = torch.full((len(data["ligand"].rgroup_mask),), -1, dtype=torch.long, device=data["ligand"].x.device)
                indices_mapping[scaf_indices] = torch.arange(len(scaf_indices), device=data["ligand"].x.device)
                
                # 从batch里单个拿出来的edge index是没有offset的，所以不用担心
                # update scaf edge index 
                old_edge_index = data["ligand", "ligand"].edge_index
                scaf_edge_mask = data["ligand"].scaf_fc_bond_mask
                updated_scaf_edge_index = indices_mapping[old_edge_index[:, scaf_edge_mask]]
                
                ## construct new complete edge index
                edges = torch.combinations(torch.arange(num_atoms, device=data["ligand"].x.device), r=2).T
                edges_rev = edges.flip(0)
                complete_edge_index = torch.cat([edges, edges_rev], dim=1)
                
                ## find edges that contain new atoms
                scaffold_edge_mask = (complete_edge_index[0] < num_scaf_atoms) & (complete_edge_index[1] < num_scaf_atoms)
                new_edges = complete_edge_index[:, ~scaffold_edge_mask]
                data["ligand", "ligand"].edge_index = torch.cat([updated_scaf_edge_index, new_edges], dim=1)
                
                # ligand bond update
                num_edges = num_atoms * (num_atoms - 1)
                ligand_num_edges.append(num_edges)
                num_scaf_edges = len(data["ligand"].fc_bond_types[data["ligand"].scaf_fc_bond_mask])
                update_num_edges = num_edges - num_scaf_edges
                data["ligand"].fc_bond_types = torch.cat([
                    data["ligand"].fc_bond_types[data["ligand"].scaf_fc_bond_mask], 
                    torch.zeros(update_num_edges, device=data["ligand"].fc_bond_types.device)
                ])
                
                # mask update
                data["ligand"].scaf_fc_bond_mask = torch.cat([
                    torch.ones(num_scaf_edges, dtype=torch.bool, device=data["ligand"].scaf_fc_bond_mask.device),
                    torch.zeros(update_num_edges, dtype=torch.bool, device=data["ligand"].scaf_fc_bond_mask.device)
                ])
                data["ligand"].rgroup_mask = torch.cat([
                    torch.zeros(num_scaf_atoms, dtype=torch.bool, device=data["ligand"].rgroup_mask.device),
                    torch.ones(update_num_atoms, dtype=torch.bool, device=data["ligand"].rgroup_mask.device)
                ])

                data_batch.append(data)
                
                assert len(data["ligand"].pos) == num_atoms and len(data["ligand"].x) == num_atoms and len(data["ligand"].rgroup_mask) == num_atoms
                rgroup_mask.append(data["ligand"].rgroup_mask)
                ligand_pos.append(data["ligand"].pos)
                ligand_v.append(data["ligand"].x.long())
                
                assert len(data["ligand"].fc_bond_types) == num_edges and len(data["ligand"].scaf_fc_bond_mask) == num_edges
                scaf_fc_bond_mask.append(data["ligand"].scaf_fc_bond_mask)
                ligand_fc_bond.append(data["ligand"].fc_bond_types)
                
            batch = Batch.from_data_list(data_batch, follow_batch=FOLLOW_BATCH)
            batch["ligand"].x = batch["ligand"].x.long()
            batch["ligand"].fc_bond_types = batch["ligand"].fc_bond_types.long()
            
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
            rgroup_mask = torch.cat(rgroup_mask, dim=0).bool()
            ligand_pos = torch.cat(ligand_pos, dim=0)
            ligand_v = torch.cat(ligand_v, dim=0).long()
            
            scaf_fc_bond_mask = torch.cat(scaf_fc_bond_mask, dim=0).bool()
            ligand_fc_bond = torch.cat(ligand_fc_bond, dim=0).long()
            # ligand_fc_bond_batch = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_edges)).to(protein_pos.device)
            ligand_fc_bond_batch = batch["ligand"].fc_bond_types_batch
            ligand_fc_edge_index = batch["ligand", "ligand"].edge_index
            
        elif sample_num_atoms == 'ref':
            ligand_pos, ligand_v, rgroup_mask, ligand_fc_bond, ligand_fc_bond_batch, scaf_fc_bond_mask = (
                batch["ligand"].pos,
                batch["ligand"].x,
                batch["ligand"].rgroup_mask,
                
                batch["ligand"].fc_bond_types,
                batch["ligand"].fc_bond_types_batch,
                batch["ligand"].scaf_fc_bond_mask,
            )
            
            ligand_fc_edge_index = batch["ligand", "ligand"].edge_index
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(ligand_pos.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])# 这个经过cunsum之后就是每个分子的起始位置
        
        # move protein center to origin & ligand correspondingly
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            rgroup_mask,
            mode=self.cfg.dynamics.center_pos_mode, # protein
        )
           
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
            scaf_fc_bond_mask=scaf_fc_bond_mask,
            ligand_fc_bond=ligand_fc_bond,
            ligand_fc_bond_batch=ligand_fc_bond_batch,
            ligand_fc_edge_index=ligand_fc_edge_index,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_graphs,
            # ligand_pos=ligand_pos,  # for debug only
            desc=desc,
        )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        # only change the predicted rgroup's position, because the original ligand pos in batch did not change CoM
        pred_pos, one_hot, one_hot_bond = final[0] + offset[batch_ligand[rgroup_mask]], final[1], final[2]
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
        pred_bond = one_hot_bond.argmax(dim=-1)
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        
        out_batch["ligand"].x[rgroup_mask] = pred_v
        out_batch["ligand"].fc_bond_types[~scaf_fc_bond_mask] = pred_bond
        
        
        # 这里的函数要改成我们16个原子的，index -> atomic number, 这个是后面用来生成分子的
        pred_atom_type = trans.get_atomic_number_from_index(
            out_batch["ligand"].x, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[int]
        

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            out_batch["ligand"].x, mode=self.cfg.data.transform.ligand_atom_mode
        ) # List[bool]

        # pred_fc_bond_types = out_batch["ligand"].fc_bond_types
        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))
        
        # add necessary dict to new pyg batch
        # out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch["ligand"].atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=ligand_pos.device)
        out_batch["ligand"].is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=ligand_pos.device)
        # # out_batch.mol = results

        _slice_dict = {
            "pos": ligand_cum_atoms,
            "x": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "is_aromatic": ligand_cum_atoms,
            # "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "pos": out_batch._inc_dict["ligand"]["pos"],
            "x": out_batch._inc_dict["ligand"]["x"],
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
