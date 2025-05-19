# This script is used to run the crossdock benchmark for the BFN model.

python train_bfn.py \
--config_file configs/default_bond.yaml \
--exp_name chembl \
--revision chembl_round6 \
--epochs 500 \
--batch_size 6 \
--bond_prediction True \
--num_samples 10 \
--resume \
--wandb_resume_id ys5gd8q0 # chembl_round6
# --wandb_resume_id dvvs1rnm # chembl_round5_bond_edge4_H800
# --wandb_resume_id 8hdwmz6c # chembl_round3_bond_edge4_H800
# --wandb_resume_id ofv3tu0p # chembl_round4_bond