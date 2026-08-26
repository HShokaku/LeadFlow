# Train the reaction-based scaffold ablation.

python train_bfn.py \
--config_file configs/Rxn_bond.yaml \
--data_name chembl_frag \
--exp_name chembl_frag \
--revision chembl_frag_round1 \
--epochs 500 \
--batch_size 6 \
--bond_prediction True \
--num_samples 10
