# Evaluate the Bemis-Murcko scaffold ablation on MolGenBench.

python train_bfn.py \
--config_file configs/bmscaffold_bond.yaml \
--data_name chembl \
--exp_name chembl_bmscaffold \
--revision chembl_bmscaffold_round1 \
--epochs 500 \
--seed 1234 \
--batch_size 75 \
--bond_prediction True \
--num_samples 200 \
--sample_num_atoms prior \
--ckpt_path ./checkpoints/LeadFlow_bmscaffold/LeadFlow_bmscaffold.ckpt \
--no_wandb \
--test_only
