# Sampling template for the BFN model.
# Run from the repository root: bash scripts/sampling_template.sh
# Each path can be absolute or relative to the repository root.  For example,
# --ckpt_path checkpoints/your_model.ckpt is a valid relative checkpoint path.

python sample_for_pocket.py \
--protein_path Path/to/protein.pdb \
--ligand_path Path/to/ligand.sdf \
--scaffold_path Path/to/scaffold.sdf \
--ckpt_path checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--sample_num_atoms prior \
--output_path Path/to/output_directory \
--batch_size 25 \
--num_samples 200 \
--seed 42 \
--devices 0 1 2 3
