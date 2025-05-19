# This script is used to run the crossdock benchmark for the BFN model.

python train_dpo.py \
--config_file configs/dpo_all_threshold5.yaml \
--exp_name chembl_dpo \
--revision round11 \
--epochs 1 \
--batch_size 3 \
--beta_reward 1 \
--dpo_beta 5 \
--lr 5e-6 \
--ft_ckpt_path ./checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--bond_prediction True \
--num_samples 200

# --resume \
# --wandb_resume_id fsecnks8  ! debug beta_reward=6.6, dpo_beta=5
