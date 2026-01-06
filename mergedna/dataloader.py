import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from typing import Dict, List, Optional

# DNA vocabulary mapping
DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


class DNADataset(Dataset):
    """
    Dataset for DNA sequences with variable lengths.
    Converts DNA base characters to token indices.
    """
    def __init__(self, sequences: List[str], max_len: Optional[int] = None):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx].strip().upper()

        # Truncate if max_len specified
        if self.max_len is not None:
            seq = seq[:self.max_len]

        # Convert to indices
        indices = [DNA_VOCAB.get(base, 0) for base in seq]
        return torch.tensor(indices, dtype=torch.long)


def collate_dna_sequences(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Collate function for variable-length DNA sequences.
    Pads sequences to the same length within a batch.
    """
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)


def sequence_to_tensor(sequence: str, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert a DNA sequence string to tensor."""
    seq = sequence.strip().upper()
    if max_len is not None:
        seq = seq[:max_len]
    indices = [DNA_VOCAB.get(base, 0) for base in seq]
    return torch.tensor(indices, dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor) -> str:
    """Convert tensor back to DNA sequence string."""
    indices = tensor.cpu().numpy()
    return ''.join([IDX_TO_BASE[idx] for idx in indices])


def create_dataloader(
    sequences: List[str],
    batch_size: int = 4,
    shuffle: bool = True,
    max_len: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:

    dataset = DNADataset(sequences, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dna_sequences,
        num_workers=num_workers
    )


def load_sequences(dir_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
        Nested dict with structure: sequences[split][label] = [seq1, seq2, ...]
    """
    sequences = {
        "train": {"positive": [], "negative": []},
        "test": {"positive": [], "negative": []}
    }

    for entry in os.scandir(dir_path):
        if not entry.is_dir():
            continue
        split_name = entry.name
        if split_name not in sequences:
            continue

        subdir_path = os.path.join(dir_path, split_name)
        for subentry in os.scandir(subdir_path):
            if not subentry.is_dir():
                continue
            label_name = subentry.name
            if label_name not in sequences[split_name]:
                continue

            label_path = os.path.join(subdir_path, label_name)
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as f:
                        sequences[split_name][label_name].append(f.read().strip())

    return sequences


dataloader = load_sequences


def merge_sequences(sequences: Dict[str, Dict[str, List[str]]]) -> List[str]:
    merged = []
    #for split_data in sequences.values():
    for label_sequences in sequences.values():
        merged.extend(label_sequences)
    return merged


def get_split_sequences(
    sequences: Dict[str, Dict[str, List[str]]],
    split: str = "train"
) -> List[str]:
    """
    Returns:
        List of sequences from the specified split
    """
    split_data = sequences.get(split, {})
    result = []
    for label_sequences in split_data.values():
        result.extend(label_sequences)
    return result


if __name__ == "__main__":

    sequences = load_sequences("data/human_nontata_promoters/")
    all_seqs = merge_sequences(sequences)
    print(f"Total sequences: {len(all_seqs)}")
    print(f"Train positive: {len(sequences['train']['positive'])}")
    print(f"Train negative: {len(sequences['train']['negative'])}")
    print(f"Example sequence: {all_seqs[0])