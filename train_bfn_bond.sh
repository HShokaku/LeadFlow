# Train the BFN model from scratch.

python train_bfn.py \
--config_file configs/default_bond.yaml \
--exp_name chembl \
--revision chembl_round6 \
--epochs 500 \
--batch_size 6 \
--bond_prediction True \
--num_samples 10
