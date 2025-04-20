import argparse
import os
import shutil
import datetime, pytz

import torch
from torch_geometric.loader import DataLoader

from src.config.config import Config
from src.models.sbdd_train_loop_bond_dpo import SBDDTrainLoopBondDPO
from src.callbacks.basic import GradientClip, NormalizerCallback
from src.callbacks.ema import EMA
from src.callbacks.validation_callback import ValidationCallback, DockingTestCallback


from src.datasets import get_datamodule
from src.datasets.pl_data import FOLLOW_BATCH
from src.data.dataset.chembl.dataset_dpo import DPODataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from absl import logging
data_module = get_datamodule("chembl", "bond")
data_module.setup("fit")


def get_dataloader(cfg):
    if 'chembl' in cfg.data.name:
        if cfg.train.bond_prediction:
            transform = 'bond'
        else:
            transform = 'simple'
            
        dataset = get_datamodule(cfg.data.name, transform, cfg.train.batch_size)
        dataset.setup('fit')
        dataset.setup('test')
        
        train_dataset = data_module.train_dataset
        dpodataset = DPODataset(
            dataset=train_dataset,
            affinity_info=cfg.data.affinity_info,
            pair_dict=cfg.data.pair_dict
        )

        train_loader = DataLoader(
            dpodataset,
            batch_size=cfg.train.batch_size,
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            follow_batch=FOLLOW_BATCH
        )
        val_loader = dataset.val_dataloader(follow_batch=FOLLOW_BATCH)
        test_loader = dataset.test_dataloader(follow_batch=FOLLOW_BATCH)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.name} not supported")
    
    return train_loader, val_loader, test_loader

def set_test_output_dir(cfg):
    path = cfg.accounting.test_outputs_dir
    version = 0
    while os.path.exists(path):
        version += 1
        path = cfg.accounting.test_outputs_dir + f'_v{version}'
    if version > 0:
        print(f'{cfg.accounting.test_outputs_dir} already exists, change test_output_dir to {path}')
    else:
        print(f'set test_output_dir as {path}')
    cfg.accounting.test_outputs_dir = path
    os.makedirs(cfg.accounting.test_outputs_dir, exist_ok=True)


def get_logger(cfg):
    os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)
    # TODO save code
    if cfg.wandb_resume_id is not None:
        wandb_logger = WandbLogger(
            id=cfg.wandb_resume_id,
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
            resume='must',
        )
    else: # start a new run
        wandb_logger = WandbLogger(
            name=f"{cfg.exp_name}_{cfg.revision}"
            + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
        )  # add wandb parameters
    return wandb_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # meta
    parser.add_argument("--config_file", type=str, default="configs/dpo.yaml",)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--data_name", type=str, default="chembl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument('--empty_folder', action='store_true')
    parser.add_argument("--test_only", action="store_true")
    
    # global config
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

    # train data params
    parser.add_argument('--random_rot', action='store_true')
    parser.add_argument("--pos_noise_std", type=float, default=0)    
    parser.add_argument("--pos_normalizer", type=float, default=2.0)    
    
    # train params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument('--v_loss_weight', type=float, default=1)
    parser.add_argument('--bond_prediction', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau'])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=str, default='Q')  # '8.0' for
    parser.add_argument('--ft_ckpt_path', type=str, default=None)
    
    # dpo params
    parser.add_argument('--beta_reward', type=float, default=6.6)
    parser.add_argument('--dpo_beta', type=float, default=5)

    # bfn params
    parser.add_argument("--sigma1_coord", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=1.5)
    parser.add_argument("--t_min", type=float, default=0.0001)
    parser.add_argument('--use_discrete_t', type=eval, default=True)
    parser.add_argument('--discrete_steps', type=int, default=1000)
    parser.add_argument('--destination_prediction', type=eval, default=True)
    parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf']) #vanilla or end_back

    parser.add_argument(
        "--time_emb_mode", type=str, default="simple", choices=["simple", "sin", 'rbf', 'rbfnn']
    )
    parser.add_argument("--time_emb_dim", type=int, default=1)
    parser.add_argument('--pos_init_mode', type=str, default='zero', choices=['zero', 'randn'])

    # eval params
    parser.add_argument('--ckpt_path', type=str, default='best', help='path to the checkpoint')
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default='ref', choices=['prior', 'ref'])
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument("--docking_mode", type=str, default="vina_score", choices=['vina_score', 'vina_dock'])


    _args = parser.parse_args()
    if _args.ckpt_path.lstrip('./') == 'checkpoints/last.ckpt':
        _args.exp_name = 'official'
        _args.revision = 'default'
    else:
        print('trying to automatically parse experiment folder...')
        try:
            *_, exp_name, revision, _, ckpt_fn = _args.ckpt_path.split('/')
            _args.exp_name = exp_name
            _args.revision = revision
            print(f'change log dir to **/{exp_name}/{revision}')
        except Exception as e:
            print(f'failed to parse experiment folder, using default exp_name and revision')
            pass

    cfg = Config(**_args.__dict__)
    if not os.path.exists(cfg.accounting.logdir):
        os.makedirs(cfg.accounting.logdir, exist_ok=True)
        # shutil.copyfile('./configs/default.yaml', cfg.accounting.dump_config_path)
    set_test_output_dir(cfg)

    if not cfg.test_only:
        seed_everything(cfg.seed, workers=True)

    logging_level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[cfg.logging_level])

    if cfg.empty_folder:
        shutil.rmtree(cfg.accounting.logdir)

    wandb_logger = get_logger(cfg)
    
    if cfg.test_only:
        if os.path.exists(cfg.accounting.dump_config_path):
            # reload training config
            tr_cfg = Config(cfg.accounting.dump_config_path)
            tr_cfg.test_only = cfg.test_only
            tr_cfg.evaluation = cfg.evaluation
            tr_cfg.visual = cfg.visual
            tr_cfg.accounting = cfg.accounting
            # TODO: temporarily test different beta1 and sigma1_coord
            tr_cfg.dynamics.beta1 = cfg.dynamics.beta1
            tr_cfg.dynamics.sigma1_coord = cfg.dynamics.sigma1_coord
            tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
            tr_cfg.train.batch_size = cfg.train.batch_size
            tr_cfg.train.bond_prediction = cfg.train.bond_prediction
            tr_cfg.data = cfg.data
            tr_cfg.seed = cfg.seed
            tr_cfg.data.name = cfg.data.name
            cfg = tr_cfg
        if not hasattr(cfg.train, 'max_grad_norm'):
            cfg.train.max_grad_norm = 'Q'
    else:
        cfg.save2yaml(cfg.accounting.dump_config_path)

    train_loader, val_loader, test_loader = get_dataloader(cfg)
    wandb_logger.log_hyperparams(cfg.todict())
    print(f"The config of this process is:\n{cfg}")

    model = SBDDTrainLoopBondDPO(config=cfg)
    
    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        # devices=[1],
        devices=[2,3,4,5,6,7], strategy="ddp_find_unused_parameters_false",
        logger=wandb_logger,
        num_sanity_val_steps=0,
        inference_mode=not cfg.test_only,
        callbacks=[
            GradientClip(max_grad_norm=cfg.train.max_grad_norm),  # time consuming
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            ValidationCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
                val_freq=cfg.train.val_freq,
                bond_prediction=cfg.train.bond_prediction,
            ),
            DockingTestCallback(
                dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
                bond_prediction=cfg.train.bond_prediction,
            ),
            ModelCheckpoint(
                monitor="val/recon_loss",
                every_n_epochs=cfg.train.ckpt_freq,
                dirpath=cfg.accounting.checkpoint_dir,
                filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}-mol_stable{val/mol_stable:.2f}-complete{val/completeness:.2f}-vina_score{val/vina_score_mean:.2f}",
                save_top_k=-1,
                mode="min",
                auto_insert_metric_name=False,
                save_last=True,
            ),
            EMA(decay=cfg.train.ema_decay),
        ],
    )
    
    if not cfg.test_only:
        # # 微调用
        # if cfg.train.ft_ckpt_path is not None:
        #     checkpoint = torch.load(cfg.train.ft_ckpt_path)
        #     model.load_state_dict(checkpoint["state_dict"])
        #     print(f"Fine-tuning from {cfg.train.ft_ckpt_path}")
        
        # 恢复训练用
        if cfg.train.resume:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt"))
        # 正常训练走这里
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # ckpt_path can be 'best', 'last', or a specific path
        trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)
    else:
        # trainer.validate(model, dataloaders=val_loader, ckpt_path=cfg.evaluation.ckpt_path)
        rank = trainer.local_rank
        seed_everything(cfg.seed + rank, workers=True)
        
        trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.evaluation.ckpt_path)