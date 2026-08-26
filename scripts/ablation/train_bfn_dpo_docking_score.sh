# Train the docking-score DPO ablation.

python train_dpo.py \
--config_file configs/dpo_all_threshold5_docking.yaml \
--data_name chembl \
--exp_name chembl_dpo_docking \
--revision round1 \
--epochs 1 \
--batch_size 3 \
--beta_reward 1 \
--dpo_beta 5 \
--lr 1e-6 \
--ft_ckpt_path ./checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--bond_prediction True \
--num_samples 200
