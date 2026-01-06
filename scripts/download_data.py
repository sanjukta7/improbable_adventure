from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import list_datasets
import os

datasets = list_datasets()
#for ds in datasets:
#    print(f"  - {ds}")

dataset_name = "human_nontata_promoters"
print(f"\nDownloading '{dataset_name}'")

dataset_path = download_dataset(dataset_name, dest_path="./data")
print(f"Dataset downloaded to: {dataset_path}")

if os.path.exists(dataset_path):
    print(f"\nDataset contents:")
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            count = len(os.listdir(item_path))
            print(f"  {item}/: {count} files")