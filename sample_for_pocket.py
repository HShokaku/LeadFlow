# import argparse
import os
import argparse

from torch_geometric.loader import DataLoader


from src.config.config import Config
# from src.models.sbdd_train_loop import SBDDTrainLoop
from src.models.sbdd_train_loop_bond import SBDDTrainLoopBond
from src.callbacks.basic import NormalizerCallback
from src.callbacks.validation_callback_for_sample import DockingTestCallback

from src.data.protein_ligand import PDBProtein
# from src.data.featurization.pl_simple import ProteinLigandSimpleFeaturization
from src.data.featurization.pl_bond import ProteinLigandBondFeaturization
from src.data.featurization.util import transfer_conformers
from src.datasets.pl_data import torchify_dict

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

import numpy as np
from rdkit import Chem


def get_input_data_template(ligand_fn, protein_fn, scaffold_fn, sample_num_atoms):
    identifier = os.path.basename(ligand_fn).split('.')[0]
    mol = Chem.MolFromMolFile(ligand_fn)
    mol = Chem.RemoveAllHs(mol)
    
    scaf_mol = Chem.MolFromMolFile(scaffold_fn)
    scaf_mol = Chem.RemoveAllHs(scaf_mol)
    
    if scaf_mol is not None and mol.HasSubstructMatch(scaf_mol):
        # TODO: add mode for denovo generation
        if mol.GetNumAtoms() == scaf_mol.GetNumAtoms():
            if sample_num_atoms == 'ref':
                raise ValueError("sample_num_atoms is 'ref' but scaffold is the same as ligand")
            elif sample_num_atoms == 'prior':
                print("Scaffold is the same as ligand")
        scaf_conformer = transfer_conformers(scaf_mol, mol)
        scaf_mol.AddConformer(scaf_conformer)
        query_mol = scaf_mol
    else:
        raise ValueError("No substructure match found")
    
    protein = PDBProtein(protein_fn)
    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(query_mol, radius=10.)
    )
    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = torchify_dict(pocket.to_dict_atom())
    
    featurizer = ProteinLigandBondFeaturization()

    data = featurizer(mol, scaf_mol, pocket_dict, identifier, mode='test')
    return data
    

def get_dataloader_from_pdb(cfg):
    assert cfg.evaluation.protein_path is not None and cfg.evaluation.ligand_path is not None
    protein_fn, ligand_fn, scaffold_fn = cfg.evaluation.protein_path, cfg.evaluation.ligand_path, cfg.evaluation.scaffold_path

    data = get_input_data_template(ligand_fn, protein_fn, scaffold_fn, cfg.evaluation.sample_num_atoms)
    test_set = [data] * cfg.evaluation.num_samples
    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(protein_fn))
    # print(f"protein root: {cfg.evaluation.docking_config.protein_root}")

    return test_loader


def call(protein_fn, ligand_fn, scaffold_fn,ckpt_path='./checkpoints/last.ckpt', output_path='./output',
         num_samples=10, sample_steps=100, sample_num_atoms='prior', devices=[0],
         beta1=1.5, sigma1_coord=0.03, sampling_strategy='end_back', seed=1234):
    
    cfg = Config('./checkpoints/config.yaml')
    seed_everything(cfg.seed)
    
    cfg.evaluation.protein_path = protein_fn
    cfg.evaluation.ligand_path = ligand_fn
    cfg.evaluation.scaffold_path = scaffold_fn
    cfg.evaluation.ckpt_path = ckpt_path
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.num_samples = num_samples
    cfg.evaluation.sample_steps = sample_steps
    cfg.evaluation.sample_num_atoms = sample_num_atoms # or 'prior'
    cfg.dynamics.beta1 = beta1
    cfg.dynamics.sigma1_coord = sigma1_coord
    cfg.dynamics.sampling_strategy = sampling_strategy
    cfg.seed = seed
    cfg.train.max_grad_norm = 'Q'
    cfg.output_path = output_path

    # print(f"The config of this process is:\n{cfg}")

    print(protein_fn, ligand_fn)
    test_loader = get_dataloader_from_pdb(cfg)
    # wandb_logger.log_hyperparams(cfg.todict())

    model = SBDDTrainLoopBond(config=cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=devices,
        # logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            DockingTestCallback(
                dataset=None,
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
                output_path=cfg.output_path,
                bond_prediction=cfg.train.bond_prediction,
            ),
        ],
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--protein_path', type=str, required=True)
    parser.add_argument('--ligand_path', type=str, required=True)
    parser.add_argument('--scaffold_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/last.ckpt')
    parser.add_argument('--sample_num_atoms', choices=['prior', 'ref'], default='prior')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.path.join('./output', os.path.basename(args.ckpt_path) + '_' + os.path.basename(args.ligand_path) + '_' + args.sample_num_atoms + '_' + str(args.num_samples))
    
    print(f"Output path: {args.output_path}")

    call(
        args.protein_path, 
        args.ligand_path,
        args.scaffold_path,
        ckpt_path=args.ckpt_path,
        sample_num_atoms=args.sample_num_atoms,
        output_path=args.output_path,
        num_samples=args.num_samples,
        devices=args.devices,
    )