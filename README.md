# LeadFlow

## 🛠️ Environment setup
```bash
mamba env create -f environment_LeadFlow_260819.yml
conda activate LeadFlow

# PoseCheck: pin the version used to calculate strain energy in this work.
git clone https://github.com/cch1999/posecheck.git
git -C posecheck checkout 57a1938
python -m pip install -e posecheck
python -m pip install -r posecheck/requirements.txt
```

## 🚀 Quick Start: Sampling

Two sampling templates are provided:

- `scripts/sampling_template.sh` uses the standard BFN checkpoint.
- `scripts/dpo_sampling_template.sh` uses the DPO checkpoint.

The sampling parameters are summarized below. Replace every `Path/to/...`
placeholder before running a template.

| Parameter | Description |
| --- | --- |
| `--protein_path` | Target protein structure in PDB format |
| `--ligand_path` | Reference ligand in SDF format in its protein-bound pose. It provides the binding coordinates used to position the scaffold within the protein structure. |
| `--scaffold_path` | Scaffold in SDF format, obtained by removing the ligand atoms or groups that should not be retained. It defines the fixed substructure to preserve during generation and must match a substructure of the reference ligand. The matched scaffold coordinates in the ligand-bound pose are used to define the surrounding protein pocket. |
| `--sample_num_atoms` | Atom-count sampling strategy. `prior` samples the number of atoms based on the sizes of the protein pocket and scaffold, and is the default and recommended setting. `ref` uses the number of atoms in the reference ligand provided by `--ligand_path`. |
| `--output_path` | Directory for generated molecules and results |
| `--batch_size` | Number of samples processed per batch |
| `--num_samples` | Total number of molecules to generate |
| `--seed` | Random seed |
| `--devices` | GPU device indices |

All relative paths are resolved from the current working directory. Run either
template from the repository root:

```bash
# Standard BFN sampling
bash scripts/sampling_template.sh

# DPO sampling
bash scripts/dpo_sampling_template.sh
```
