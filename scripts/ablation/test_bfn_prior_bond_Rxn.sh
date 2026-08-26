# Evaluate the reaction-based scaffold ablation on MolGenBench.

python train_bfn.py \
--config_file configs/Rxn_bond.yaml \
--data_name chembl \
--exp_name chembl_frag \
--revision chembl_frag_round1 \
--epochs 500 \
--seed 5678 \
--batch_size 90 \
--bond_prediction True \
--num_samples 200 \
--sample_num_atoms prior \
--ckpt_path ./checkpoints/LeadFlow_Rxn/LeadFlow_Rxn.ckpt \
--no_wandb \
--test_only
