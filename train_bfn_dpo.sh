# Train the DPO model from the pretrained BFN checkpoint.

python train_dpo.py \
--config_file configs/dpo_all_threshold5.yaml \
--exp_name chembl_dpo \
--revision round12 \
--epochs 1 \
--batch_size 3 \
--beta_reward 1 \
--dpo_beta 5 \
--lr 1e-6 \
--ft_ckpt_path ./checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--bond_prediction True \
--num_samples 200
