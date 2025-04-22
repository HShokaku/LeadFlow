# This script is used to run the crossdock benchmark for the BFN model.

python train_dpo.py \
--config_file configs/dpo.yaml \
--exp_name chembl_dpo \
--revision round1 \
--epochs 100 \
--batch_size 3 \
--beta_reward 1 \
--dpo_beta 5 \
--ft_ckpt_path ./checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--bond_prediction True \
--num_samples 10

# --resume \
# --wandb_resume_id fsecnks8  ! debug beta_reward=6.6, dpo_beta=5
