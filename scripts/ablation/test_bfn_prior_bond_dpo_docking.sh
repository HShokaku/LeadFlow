# Evaluate the docking-score DPO ablation on MolGenBench.

python train_dpo.py \
--config_file configs/dpo_all_threshold5_docking.yaml \
--data_name chembl \
--exp_name chembl_dpo_docking \
--revision round1 \
--epochs 500 \
--batch_size 100 \
--bond_prediction True \
--num_samples 200 \
--seed 2666 \
--sample_num_atoms prior \
--ckpt_path ./checkpoints/LeadFlow_dpo_docking/LeadFlow_dpo_docking.ckpt \
--no_wandb \
--test_only
