import argparse
import math
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from rdkit import Chem

from src.utils import disable_obabel_and_rdkit_logging
from src.datasets import get_datamodule


def _is_scaffold_conditioned(row):
    """Return whether a ligand has a valid partial scaffold for DPO training."""
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None:
        raise ValueError(f'Could not parse SMILES for {row["identifier"]}')
    mol = Chem.RemoveAllHs(mol)

    scaffold = None
    if not pd.isna(row["Scaffold"]):
        scaffold = Chem.MolFromSmarts(row["Scaffold"])

    if scaffold is None:
        return False

    matches = mol.GetSubstructMatches(scaffold)
    return bool(matches) and len(matches[0]) < mol.GetNumAtoms()


def create_dpo_pickles(train_csv, output_dir=None, threshold=5.0):
    """Create the affinity metadata and DPO pair dictionary from train.csv."""
    train_csv = Path(train_csv)
    output_dir = Path(output_dir) if output_dir is not None else train_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    required_columns = {"identifier", "sries_id", "affinity", "smiles", "Scaffold"}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {train_csv}: {missing}")

    # Affinity values are reported in nM. Only positive values can be converted
    # to pAffinity and used to construct preference pairs.
    train_df = train_df[train_df["affinity"] > 0].copy()
    train_df["pAffinity"] = train_df["affinity"].apply(
        lambda value: -math.log10(value * 1e-9)
    )

    affinity_info = train_df.set_index("identifier").to_dict(orient="index")
    affinity_path = output_dir / "affinity_info_all.pkl"
    with affinity_path.open("wb") as f:
        pickle.dump(affinity_info, f)

    # Keep only scaffold-conditioned samples for DPO training.
    scaffold_conditioned = train_df.apply(_is_scaffold_conditioned, axis=1)
    pair_df = train_df[scaffold_conditioned]
    pair_df = pair_df.groupby("sries_id").filter(lambda group: len(group) >= 2)

    pair_dict = defaultdict(list)
    for sries_id, group in pair_df.groupby("sries_id"):
        records = list(
            group[["affinity", "identifier"]].itertuples(index=False, name=None)
        )
        for (affinity_1, identifier_1), (affinity_2, identifier_2) in combinations(
            records, 2
        ):
            # A lower measured affinity value indicates the more active ligand.
            if affinity_1 >= threshold * affinity_2:
                pair_dict[sries_id].append((identifier_2, identifier_1))
            elif affinity_2 >= threshold * affinity_1:
                pair_dict[sries_id].append((identifier_1, identifier_2))

    pair_path = output_dir / f"dpo_pair_dict_threshold{threshold:g}.pkl"
    with pair_path.open("wb") as f:
        pickle.dump(pair_dict, f)

    print(f"Saved {len(affinity_info)} affinity records to {affinity_path}")
    print(
        f"Saved {sum(len(pairs) for pairs in pair_dict.values())} DPO pairs "
        f"from {len(pair_dict)} series to {pair_path}"
    )
    return affinity_path, pair_path


if __name__ == '__main__':
    disable_obabel_and_rdkit_logging()
    parser = argparse.ArgumentParser(
        prog="create_dataset.py",
        description="Create pretrain dataset with given name",
        epilog="Example: python create_dataset.py SureChEMBL",
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--transform", type=str, default="bond", help="Transform to use")
    parser.add_argument("--stage", type=str, default="test", choices=["fit", "test"], help="Stage to setup")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    dataset_root = Path("data") / args.dataset_name
    data_module = get_datamodule(args.dataset_name, args.transform)

    print(f"Creating dataset {args.dataset_name} with pre_transform {data_module.pre_transform} and pre_filter {data_module.pre_filter}")
    data_module.setup(args.stage)

    if args.dataset_name == "chembl" and args.stage == "fit":
        create_dpo_pickles(dataset_root / "raw" / "train.csv")
