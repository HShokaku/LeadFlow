from absl import logging

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import Struct
from src.models.common import compose_context_mr_bond, ShiftedSoftplus
from src.models.bfn_base import BFNBase
from src.models.uni_transformer_edge import UniTransformerO2TwoUpdateGeneralBond


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RBF(nn.Module):
    def __init__(self, start, end, n_center):
        super().__init__()
        self.start = start
        self.end = end
        self.n_center = n_center
        self.centers = torch.linspace(start, end, n_center)
        self.width = (end - start) / n_center

    def forward(self, x):
        assert x.ndim >= 2
        out = (x - self.centers.to(x.device)) / self.width
        ret = torch.exp(-0.5 * out**2)
        return F.normalize(ret, dim=-1, p=1) * 2 - 1


class TimeEmbedLayer(nn.Module):
    def __init__(self, time_emb_mode, time_emb_dim):
        super().__init__()
        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim

        if self.time_emb_mode == "simple":
            assert self.time_emb_dim == 1
            self.time_emb = lambda x: x
        elif self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        elif self.time_emb_mode == "rbf":
            self.time_emb = RBF(0, 1, self.time_emb_dim)
        elif self.time_emb_mode == "rbfnn":
            self.time_emb = nn.Sequential(
                RBF(0, 1, self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t):
        return self.time_emb(t)


class BFN4SBDDScoreModel(BFNBase):
    def __init__(
        self,
        # in_node_nf,
        # hidden_nf=64,
        net_config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        ligand_bond_feature_dim,
        device="cuda",
        condition_time=True,
        sigma1_coord=0.02,
        beta1=3.0,
        use_discrete_t=False,
        discrete_steps=1000,
        t_min=0.0001,
        # no_diff_coord=False,
        node_indicator=True,
        # charge_discretised_loss = False
        time_emb_mode='simple',
        time_emb_dim=1,
        center_pos_mode='protein',
        pos_init_mode='zero',
        destination_prediction = False,
        sampling_strategy = "vanilla"
    ):
        super(BFN4SBDDScoreModel, self).__init__()
        # if include_charge:
        #     out_node_nf = in_node_nf + 2
        # else:
        #     out_node_nf = in_node_nf + 1
        net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == 'unio2net_bond':
            self.unio2net = UniTransformerO2TwoUpdateGeneralBond(**net_config.todict())
        else:
            raise NotImplementedError
        
        self.hidden_dim = net_config.hidden_dim
        self.num_atom_classes = ligand_atom_feature_dim
        self.num_bond_classes = ligand_bond_feature_dim
        
        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 3
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        # TODO： add scaffold atom embedding
        self.scaffold_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)
        # bond embedding without indicator
        self.scaffold_bond_emb = nn.Linear(ligand_bond_feature_dim, self.hidden_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein'， scaffold] 

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )
        self.ligand_bond_emb = nn.Linear(
            ligand_bond_feature_dim + self.time_emb_dim, self.hidden_dim
        )

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]
        self.bond_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_bond_feature_dim),
        )

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)  # coordinate sigma1, a schedule for bfn
        self.beta1_atom = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.beta1_bond = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy

    def interdependency_modeling(
        self,
        time,
        protein_pos,  # transform from the orginal BFN codebase
        protein_v,  # transform from
        batch_protein,  # index for protein
        scaffold_pos,
        scaffold_v,
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        gamma_coord,
        rgroup_mask,
        # anchor_mask,
        theta_h_bond_t,
        ligand_fc_bond,
        scaf_fc_bond_mask,
        ligand_fc_bond_batch,
        ligand_fc_edge_index,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
    ):
        """
        Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits).
        Draw output_sample = x' ~ p_O (x' | θ; t).
            continuous x ~ δ(x - x_hat(θ, t))
            discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [node_num x batch_size, protein_atom_feature_dim] := [N_protein, 27]
            batch_protein: [node_num x batch_size] := [N_protein]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]
        theta_h_bond_t = 2 * theta_h_bond_t - 1 

        init_ligand_v = theta_h_t
        init_ligand_b = theta_h_bond_t
        
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0: # 1 for simple
            time_emb = self.time_emb_layer(time)
            time_emb_atom = time_emb[:len(theta_h_t)]
            time_emb_bond = time_emb[len(theta_h_t):]
            
            input_ligand_feat = torch.cat([init_ligand_v, time_emb_atom], -1)
            input_bond_feat = torch.cat([init_ligand_b, time_emb_bond], -1)
        else:
            input_ligand_feat = init_ligand_v
            input_bond_feat = init_ligand_b

        h_protein = self.protein_atom_emb(protein_v)  # [N_protein, self.hidden_dim - 1]
        h_scaffold = self.scaffold_atom_emb(scaffold_v) # [N_scaffold, self.hidden_dim - 1]
        h_scaffold_bond = self.scaffold_bond_emb(ligand_fc_bond[scaf_fc_bond_mask])
        
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_rgroup self.hidden_dim - 1]
        init_ligand_h_bond = self.ligand_bond_emb(input_bond_feat)
        # init_ligand_h = input_ligand_feat # TODO: no embedding for ligand atoms, check whether this make sense.
        
        h_bond = torch.cat([h_scaffold_bond, init_ligand_h_bond], dim=0)  # [N_protein + N_scaffold + N_rgroup, self.hidden_dim - 1]
        h_bond[scaf_fc_bond_mask] = h_scaffold_bond
        h_bond[~scaf_fc_bond_mask] = init_ligand_h_bond
        
        if self.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.tensor([1, 0, 0]).unsqueeze(0).repeat(len(h_protein), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim ] # 0 for indicator
            h_scaffold = torch.cat(
                [h_scaffold, torch.tensor([0, 1, 0]).unsqueeze(0).repeat(len(h_scaffold), 1).to(h_protein)], -1
            )  # 2 for indicator
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.tensor([0, 0, 1]).unsqueeze(0).repeat(len(init_ligand_h), 1).to(h_protein)], -1
            )  # [N_ligand, self.hidden_dim] # 1 for indicator
        
        
        # mask_rgroup contain protein, scaffold, rgroup, rgroup_mask only contain rgroup and scaffold
        h_all, pos_all, batch_all, mask_rgroup, mask_scaffold, ligand_idx_in_ctx = compose_context_mr_bond(
            h_protein=h_protein,
            h_scaffold=h_scaffold,
            h_rgroup=init_ligand_h,
            pos_protein=protein_pos,
            pos_scaffold=scaffold_pos,
            pos_rgroup=mu_pos_t,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            rgroup_mask=rgroup_mask,
        )
        
        bond_index_in_all = ligand_idx_in_ctx[ligand_fc_edge_index]
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)
        # ---------------------

        # time = 2 * time - 1
        outputs = self.unio2net(
            h_all, 
            pos_all,
            h_bond,
            bond_index_in_all,
            mask_rgroup, 
            mask_scaffold, 
            batch_all, 
            return_all=return_all, 
            # fix_x=fix_x
        )
        final_pos, final_h, final_h_bond = (
            outputs["x"],
            outputs["h"],
            outputs["h_bond"],
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h, final_rgroup_h_bond = final_pos[mask_rgroup], final_h[mask_rgroup], final_h_bond[~scaf_fc_bond_mask]
        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]
        final_rgroup_bond = self.bond_inference(final_rgroup_h_bond)

        # TODO: think about equivariance for pos & center of mass
        # final_ligand_pos = final_ligand_pos - mu_pos_t  # model the delta

        # _, final_ligand_pos, _ = center_pos(
        #     protein_pos, final_ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # 1. for continuous, network outputs eps_hat(θ, t)
        # Eq.(84): x_hat(θ, t) = μ / γ(t) − \sqrt{(1 − γ(t)) / γ(t)} * eps_hat(θ, t)
        if not self.destination_prediction:
            coord_pred = (
                mu_pos_t / gamma_coord
                - torch.sqrt((1 - gamma_coord) / gamma_coord) * final_ligand_pos
            )
            coord_pred = torch.where(
                time < self.t_min, torch.zeros_like(mu_pos_t), coord_pred
            )
        else:
            coord_pred = final_ligand_pos #add destination prediction. 

        # if self.condition_time:
        #     # Slice off last dimension which represented time.
        #     h_final = h_final[:, :-1]

        # 2. for discrete, network outputs Ψ(θ, t)
        # take softmax will do
        p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)
        p0_h_bond = torch.nn.functional.softmax(final_rgroup_bond, dim=-1)
        
        return coord_pred, p0_h, p0_h_bond

    def reconstruction_loss_one_step(
        self,
        t_node, 
        t_bond,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        rgroup_mask,
        scaf_fc_bond_mask,
        ligand_fc_bond,
        ligand_fc_bond_batch,
        ligand_fc_edge_index,
    ):
        # TODO: implement reconstruction loss (but do we really need it?)
        return self.loss_one_step(
            t_node, 
            t_bond,
            protein_pos,
            protein_v,
            batch_protein,
            ligand_pos,
            ligand_v,
            batch_ligand,
            rgroup_mask,
            scaf_fc_bond_mask,
            ligand_fc_bond,
            ligand_fc_bond_batch,
            ligand_fc_edge_index,
        )

    def loss_one_step(
        self,
        t_node,  # [N_rgroup, 1]
        t_bond,  # [N_bond_no_scaf, 1]
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        rgroup_mask,
        scaf_fc_bond_mask,
        ligand_fc_bond,
        ligand_fc_bond_batch,
        ligand_fc_edge_index,
        # anchor_mask,
    ):
        K_atom = self.num_atom_classes
        K_bond = self.num_bond_classes
        
        assert ligand_v.max().item() < K_atom, f"Error: {ligand_v.max().item()} >= {K_atom}"
        assert ligand_fc_bond.max().item() < K_bond, f"Error: {ligand_fc_bond.max().item()} >= {K_bond}"
        ligand_v = F.one_hot(ligand_v, K_atom).float()  # [N, K_atom]
        ligand_fc_bond = F.one_hot(ligand_fc_bond, K_bond).float()  # [N, K_bond]
        
        rgroup_pos = ligand_pos[rgroup_mask]
        rgroup_v = ligand_v[rgroup_mask]
        rgroup_bond = ligand_fc_bond[~scaf_fc_bond_mask]
        
        scaffold_pos = ligand_pos[~rgroup_mask]
        scaffold_v = ligand_v[~rgroup_mask]
        # scaffold_bond = ligand_fc_bond[scaf_fc_bond_mask]

        # 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
        # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t_node, sigma1=self.sigma1_coord, x=rgroup_pos
        )  # [N, 3], [N, 1]

        # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
        theta_atom = self.discrete_var_bayesian_update(
            t_node, beta1=self.beta1_atom, x=rgroup_v, K=K_atom
        )  # [N, K]
        # TODO:change beta for bond
        theta_bond = self.discrete_var_bayesian_update(
            t_bond, beta1=self.beta1_bond, x=rgroup_bond, K=K_bond
        )
        
        ligand_fc_bond[~scaf_fc_bond_mask] = theta_bond
        
        # 2. Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits)
        # continuous x ~ δ(x − x_hat(θ, t))
        # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        coord_pred, p0_h, p0_h_bond = self.interdependency_modeling(
            time=torch.cat([t_node, t_bond], dim=0),
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            scaffold_pos=scaffold_pos,
            scaffold_v=scaffold_v,
            theta_h_t=theta_atom,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
            rgroup_mask=rgroup_mask,
            theta_h_bond_t=theta_bond,
            ligand_fc_bond=ligand_fc_bond,
            scaf_fc_bond_mask=scaf_fc_bond_mask,
            ligand_fc_bond_batch=ligand_fc_bond_batch,
            ligand_fc_edge_index=ligand_fc_edge_index,
            # anchor_mask=anchor_mask,
        )  # [N, 3], [N, K], [?]

        # if self.include_charge:
        #     k_c = torch.tensor(self.k_c).to(self.device).unsqueeze(-1).unsqueeze(0)
        #     k_hat = (k_hat * k_c).sum(dim=1)
        # average
        # print("x",x.shape,"p0_h",p0_h.shape,"k_hat",k_hat.shape,"charges",charges.shape,mu_charge.shape)

        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t_node,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=rgroup_pos,
                segment_ids=batch_ligand[rgroup_mask],
            )  # [B,]
            dloss_atom = self.ctime4discrete_loss(
                t=t_node,
                beta1=self.beta1_atom,
                one_hot_x=rgroup_v,
                p_0=p0_h,
                K=K_atom,
                segment_ids=batch_ligand[rgroup_mask],
            )  # [B,]
            dloss_bond = self.ctime4discrete_loss(
                t=t_bond,
                beta1=self.beta1_bond,
                one_hot_x=rgroup_bond,
                p_0=p0_h_bond,
                K=K_bond,
                segment_ids=ligand_fc_bond_batch[~scaf_fc_bond_mask],
            )
            
        else:
            i_node = (t_node * self.discrete_steps).int() + 1  # discrete interval [1,N]
            closs = self.dtime4continuous_loss(
                i=i_node,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=rgroup_pos,
                segment_ids=batch_ligand[rgroup_mask],
            )

            # closs = self.ctime4continuous_loss(
            #     t=t, sigma1=self.sigma1_coord, x_pred=coord_pred, x=ligand_pos, segment_ids=batch_ligand
            # )  # [B,]
            # dloss = self.ctime4discrete_loss(
            #     t=t, beta1=self.beta1, one_hot_x=ligand_v, p_0=p0_h, K=K, segment_ids=batch_ligand
            # )  # [B,]

            # TODO: change to dtime4discrete_loss
            dloss_atom = self.dtime4discrete_loss_prob(
                i=i_node,
                N=self.discrete_steps,
                beta1=self.beta1_atom,
                one_hot_x=rgroup_v,
                p_0=p0_h,
                K=K_atom,
                segment_ids=batch_ligand[rgroup_mask],
            )
            
            i_bond = (t_bond * self.discrete_steps).int() + 1
            dloss_bond = self.dtime4discrete_loss_prob(
                i=i_bond,
                N=self.discrete_steps,
                beta1=self.beta1_bond,
                one_hot_x=rgroup_bond,
                p_0=p0_h_bond,
                K=K_bond,
                segment_ids=ligand_fc_bond_batch[~scaf_fc_bond_mask],
            )

            # mixed loss
            # dloss = self.ctime4discrete_loss(
            #     t=t,
            #     beta1=self.beta1,
            #     one_hot_x=ligand_v,
            #     p_0=p0_h,
            #     K=K,
            #     segment_ids=batch_ligand,
            # )  # [B,]

        # closs = self.ctime4continuous_loss(
        #     t=t, sigma1=self.sigma1_coord, x_pred=coord_pred, x=ligand_pos
        # )
        # dloss = self.ctime4discrete_loss(
        #     t=t, beta1=self.beta1, one_hot_x=ligand_v, p_0=p0_h, K=K
        # )
        # TODO: check compatible with charge
        # if self.include_charge:
        #     if self.charge_discretised_loss:
        #         discretized_loss = self.ctime4discreteised_loss(
        #         t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges)
        #     else:
        #         discretized_loss = self.ctime4continuous_loss(
        #         t=t, sigma1=self.sigma1_charges, x_pred=k_hat, x=charges)
        # else:
        # discretized_loss = torch.zeros_like(closs)
        
        return closs, dloss_atom, dloss_bond

    def sample(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        rgroup_mask,
        scaf_fc_bond_mask,
        ligand_fc_bond,
        ligand_fc_bond_batch,
        ligand_fc_edge_index,
        # anchor_mask,
        n_nodes,  # B
        sample_steps=1000,
        desc='',
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """

        # 1. Initialize prior input parameters θ for p_I(x | θ_0),
        # for continuous, θ_0 = N(0, I)
        # for discrete, θ_0 = 1/K ∈ [0,1]**(KD)
        K_atom = self.num_atom_classes
        K_bond = self.num_bond_classes
        
        ligand_v = F.one_hot(ligand_v, K_atom).float()  # [
        ligand_fc_bond = F.one_hot(ligand_fc_bond, K_bond).float()  # [N, K_bond]
        
        scaffold_pos = ligand_pos[~rgroup_mask]
        scaffold_v = ligand_v[~rgroup_mask]
        
        
        if self.pos_init_mode == 'zero':
            mu_pos_t = torch.zeros((n_nodes, 3)).to(
                self.device
            )  # [N, 3] coordinates prior N(0, 1)
        elif self.pos_init_mode == 'randn':
            mu_pos_t = torch.randn((n_nodes, 3)).to(self.device)

        theta_h_t = (
            torch.ones((n_nodes, K_atom)).to(self.device) / K_atom
        )  # [N, K] discrete prior (uniform 1/K)
        theta_h_bond_t = (
            torch.ones((n_nodes, K_bond)).to(self.device) / K_bond
        )
        ro_coord = 1

        sample_traj = []
        theta_traj = []
        y_traj = []

        mu_pos_t = mu_pos_t[batch_ligand[rgroup_mask]]
        theta_h_t = theta_h_t[batch_ligand[rgroup_mask]]
        theta_h_bond_t = theta_h_bond_t[ligand_fc_bond_batch[~scaf_fc_bond_mask]]
        
        for i in trange(1, sample_steps + 1, desc=f'{desc}'):
            t = torch.ones((n_nodes, 1)).to(self.device) * (i - 1) / sample_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            t_node = t[batch_ligand[rgroup_mask]]
            t_bond = t[ligand_fc_bond_batch[~scaf_fc_bond_mask]]
            # Eq.(84): γ(t) = σ1^(2t)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t_node)
            # gamma_charge = 1 - torch.pow(self.sigma1_charges, 2 * t)

            # 2. Compute output distribution parameters for p_O (x' | θ; t)
            # output_params = Ψ(θ,t), x_hat or e_hat for output distribution
            # continuous x ~ δ(x − x_hat(θ, t))
            # discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k

            # debug only
            # mu_coord_gt, gamma_coord_gt = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=ligand_pos)

            coord_pred, p0_h_pred, p0_h_bond_pred = self.interdependency_modeling(
                time=torch.cat([t_node, t_bond], dim=0),
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                scaffold_pos=scaffold_pos,
                scaffold_v=scaffold_v,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_t, 
                # mu_pos_t=mu_coord_gt,  # fix mu pos guidance, type decoding
                # mu_pos_t=mu_pos_t if i > sample_steps/10 else mu_coord_gt,  # early guidance
                mu_pos_t=mu_pos_t,  # no guidance
                gamma_coord=gamma_coord,
                rgroup_mask=rgroup_mask,
                theta_h_bond_t=theta_h_bond_t,
                ligand_fc_bond=ligand_fc_bond,
                scaf_fc_bond_mask=scaf_fc_bond_mask,
                ligand_fc_bond_batch=ligand_fc_bond_batch,
                ligand_fc_edge_index=ligand_fc_edge_index,
                # anchor_mask=anchor_mask,
                # TODO: add charge
                # mu_charge_t=mu_charge_t,
                # gamma_charge=gamma_charge,
            )

            # debug only
            # if i % (sample_steps // 10) == 0:
            #     print("\nt*1000", t[0,0].item()*1000, "gamma_coord", gamma_coord[0,0].item(), torch.sqrt(gamma_coord * (1 - gamma_coord))[0,0].item())
            #     print("ligand_pos", ligand_pos)
            #     print("mu_coord_gt", mu_coord_gt)
            #     assert torch.allclose(gamma_coord_gt, gamma_coord, atol=1e-4), "gamma_coord_gt != gamma_coord"
            #     print("mu_pos_t", mu_pos_t)
            #     print("coord_pred", coord_pred)

            # maintain theta_traj
            theta_traj.append((mu_pos_t, theta_h_t, theta_h_bond_t))
            # TODO delete the following condition
            if not torch.all(p0_h_pred.isfinite()):
                p0_h_pred = torch.where(
                    p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred)
                )
                logging.warn("p0_h_pred is not finite")
            if not torch.all(p0_h_bond_pred.isfinite()):
                p0_h_bond_pred = torch.where(
                    p0_h_bond_pred.isfinite(), p0_h_bond_pred, torch.zeros_like(p0_h_bond_pred)
                )
                logging.warn("p0_h_bond_pred is not finite")

            p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)
            sample_pred = torch.distributions.Categorical(p0_h_pred).sample()
            sample_pred = F.one_hot(sample_pred, num_classes=K_atom)
            
            p0_h_bond_pred = torch.clamp(p0_h_bond_pred, min=1e-6)
            sample_pred_bond = torch.distributions.Categorical(p0_h_bond_pred).sample()
            sample_pred_bond = F.one_hot(sample_pred_bond, num_classes=K_bond)

            # if self.include_charge:
            #     sample_traj.append((coord_pred[:, :-1], sample_pred))
            # else:
            #     sample_traj.append((coord_pred, sample_pred))

            # 3. Model sender distribution for sample y ~ p_S (y | x'; α)
            # Algorithm (3)
            # for continuous, y.shape == data.shape
            # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )
            # Eq.(86): p_S (y | x'; α) = N(y | x', 1/α*I)
            # (meaning that y ∼ p_R(· | θ_{i−1}; t_{i−1}, α_i) — see Eq. 4)
            y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(
                1 / alpha_coord
            )
            # Algorithm (9)
            # for discrete, y \in R^K, while data \in {1,K}, cf. Eq.(141)
            # where e_k is network output p0_h_pred
            # Eq.(193): α_i = β(1) * (2i − 1) / n**2
            alpha_h = self.beta1_atom * (2 * i - 1) / (sample_steps**2)
            alpha_h_bond = self.beta1_bond * (2 * i - 1) / (sample_steps**2)
            
            def get_y_pred(p0, alpha_h, K):
                k = torch.distributions.Categorical(probs=p0).sample()
                e_k = F.one_hot(k, num_classes=K).float()
                # y ~ N(α(Ke_k − 1) , αKI)
                mean = alpha_h * (K * e_k - 1)
                var = alpha_h * K
                std = torch.full_like(mean, fill_value=var).sqrt()
                y_pred = mean + std * torch.randn_like(e_k)
                return y_pred
            y_h = get_y_pred(p0_h_pred, alpha_h, K_atom)
            y_h_bond = get_y_pred(p0_h_bond_pred, alpha_h_bond, K_bond)
                
            y_traj.append((y_coord, y_h, y_h_bond))

            if self.sampling_strategy == "vanilla":
                sample_traj.append((coord_pred, sample_pred, sample_pred_bond))

                # 4. Bayesian update input parameters θ_i = h(θ_{i−1}, y) for p_I(x | θ_i; t_i)
                # for continuous, Eq.(49): ρi = ρ_{i−1} + α,
                # Eq.(50): μi = [μ_{i−1}ρ_{i−1} + yα] / ρi
                mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
                    ro_coord + alpha_coord
                )
                ro_coord = ro_coord + alpha_coord

                # for discrete, Eq.(171): h(θi−1, y, α) := e**y * θ_{i−1} / \sum_{k=1}^K e**y_k (θ_{i−1})_k
                theta_prime = torch.exp(y_h) * theta_h_t  # e^y * θ_{i−1}
                theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)
                
                theta_prime_bond = torch.exp(y_h_bond) * theta_h_bond_t
                theta_h_bond_t = theta_prime_bond / theta_prime_bond.sum(dim=-1, keepdim=True)

            elif "end_back" in self.sampling_strategy:
                t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                t_node = t[batch_ligand[rgroup_mask]]
                t_bond = t[ligand_fc_bond_batch[~scaf_fc_bond_mask]]
                if self.sampling_strategy == "end_back":
                    theta_h_t = self.discrete_var_bayesian_update(t_node, beta1=self.beta1_atom, x=sample_pred, K=K_atom)
                    theta_h_bond_t = self.discrete_var_bayesian_update(t_bond, beta1=self.beta1_bond, x=sample_pred_bond, K=K_bond)
                elif self.sampling_strategy == "end_back_pmf":
                    theta_h_t = self.discrete_var_bayesian_update(t_node, beta1=self.beta1_atom, x=p0_h_pred, K=K_atom)
                    theta_h_bond_t = self.discrete_var_bayesian_update(t_bond, beta1=self.beta1_bond, x=p0_h_bond_pred, K=K_bond)
                elif self.sampling_strategy == "end_back_mode":
                    p0_mode = torch.argmax(p0_h_pred, dim=-1)
                    mode_pred = F.one_hot(p0_mode, num_classes=K_atom).float()
                    theta_h_t = self.discrete_var_bayesian_update(t_node, beta1=self.beta1_atom, x=mode_pred, K=K_atom)
                    
                    p0_mode_bond = torch.argmax(p0_h_bond_pred, dim=-1)
                    mode_pred_bond = F.one_hot(p0_mode_bond, num_classes=K_bond).float()
                    theta_h_bond_t = self.discrete_var_bayesian_update(t_bond, beta1=self.beta1_bond, x=mode_pred_bond, K=K_bond)
                else:
                    raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
                mu_pos_t = self.continuous_var_bayesian_update(t_node, sigma1=self.sigma1_coord, x=coord_pred)[0]
                sample_traj.append((coord_pred, sample_pred, sample_pred_bond))

            else:
                raise NotImplementedError
            
            

        # 5. Compute final output distribution parameters for p_O (x' | θ; t)
        mu_pos_final, p0_h_final, p0_h_bond_final = self.interdependency_modeling(
            time=torch.ones((len(t_node)+len(t_bond), 1)).to(self.device),
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            scaffold_pos=scaffold_pos,
            scaffold_v=scaffold_v,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_t,
            theta_h_bond_t=theta_h_bond_t,
            mu_pos_t=mu_pos_t,
            # mu_charge_t=mu_charge_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            rgroup_mask=rgroup_mask,
            ligand_fc_bond=ligand_fc_bond,
            scaf_fc_bond_mask=scaf_fc_bond_mask,
            ligand_fc_bond_batch=ligand_fc_bond_batch,
            ligand_fc_edge_index=ligand_fc_edge_index,
            # anchor_mask=anchor_mask,
            # gamma_charge=1 - self.sigma1_charges**2,
        )
        # TODO delete the following condition
        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(
                p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final)
            )
            logging.warn("p0_h_pred is not finite")
        if not torch.all(p0_h_bond_final.isfinite()):
            p0_h_bond_final = torch.where(
                p0_h_bond_final.isfinite(), p0_h_bond_final, torch.zeros_like(p0_h_bond_final)
            )
            logging.warn("p0_h_bond_pred is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        p0_h_bond_final = torch.clamp(p0_h_bond_final, min=1e-6)
        theta_traj.append((mu_pos_final, p0_h_final, p0_h_bond_final))

        # 6. Draw final sample from p_O (· | θ_n, 1)
        # Update: directly take the mode of categorical distribution (as done in BFN paper)
        k_final = p0_h_final
        k_bond_final = p0_h_bond_final
        # k_final = torch.distributions.Categorical(p0_h_final).sample()
        # k_final = F.one_hot(k_final, num_classes=K)

        sample_traj.append((mu_pos_final, k_final, k_bond_final))

        return theta_traj, sample_traj, y_traj
