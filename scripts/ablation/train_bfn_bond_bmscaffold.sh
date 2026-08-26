# Train the Bemis-Murcko scaffold ablation.

python train_bfn.py \
--config_file configs/bmscaffold_bond.yaml \
--data_name chembl_bmscaffold \
--exp_name chembl_bmscaffold \
--revision chembl_bmscaffold_round1 \
--epochs 500 \
--batch_size 6 \
--bond_prediction True \
--num_samples 10
