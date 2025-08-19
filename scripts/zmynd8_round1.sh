# This script is used to run the crossdock benchmark for the BFN model.

ROUND=round1

python sample_for_pocket.py \
--protein_path /home/fanzhehuan/myprojects/BFNDec_chembl/case_study/zmynd8/zmynd8_protein_AF3.pdb \
--ligand_path /home/fanzhehuan/myprojects/BFNDec_chembl/case_study/zmynd8/$ROUND/zmynd8_ligand_AF3.sdf \
--scaffold_path /home/fanzhehuan/myprojects/BFNDec_chembl/case_study/zmynd8/$ROUND/zmynd8_scaf_AF3.sdf \
--ckpt_path /home/fanzhehuan/myprojects/BFNDec_chembl/checkpoints/chembl_round6_epoch352/chembl_round6_epoch352.ckpt \
--sample_num_atoms prior \
--output_path /home/fanzhehuan/myprojects/BFNDec_chembl/case_study/zmynd8/$ROUND/sample_output \
--batch_size 25 \
--num_samples 10000 \
--devices 0 1 2 3 4 5 6 7