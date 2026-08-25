# Evaluate the pretrained DPO model on the test set.

python train_dpo.py \
--config_file configs/dpo_all_threshold5.yaml \
--exp_name chembl_dpo \
--revision round12 \
--epochs 500 \
--batch_size 100 \
--bond_prediction True \
--num_samples 200 \
--sample_num_atoms prior \
--ckpt_path ./checkpoints/dpo_round12/chembl_dpo_round12.ckpt \
--test_only
