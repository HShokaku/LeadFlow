# Evaluate the pretrained BFN model on the test set.

python train_bfn.py \
--config_file configs/default_bond.yaml \
--exp_name chembl \
--revision chembl_round6 \
--epochs 500 \
--batch_size 75 \
--bond_prediction True \
--num_samples 200 \
--sample_num_atoms prior \
--ckpt_path ./checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--test_only
