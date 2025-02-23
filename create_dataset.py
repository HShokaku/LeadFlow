import argparse

from src.utils import disable_obabel_and_rdkit_logging
from src.datasets import get_datamodule

if __name__ == '__main__':
    disable_obabel_and_rdkit_logging()
    parser = argparse.ArgumentParser(
        prog="create_dataset.py",
        description="Create pretrain dataset with given name",
        epilog="Example: python create_dataset.py SureChEMBL",
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--transform", type=str, default="simple", help="Transform to use")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    data_module = get_datamodule(args.dataset_name, args.transform)
    
    print(f"Creating dataset {args.dataset_name} with pre_transform {data_module.pre_transform} and pre_filter {data_module.pre_filter}")
    data_module.setup("fit")
    data_module.setup("test")