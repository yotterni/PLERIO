from plerio.data_utils.kmer_counting import KmerCounter

from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch
import pytorch_lightning as pl


class KmerDataset(Dataset):

    def __init__(
        self,
        data_path: str | Path,
        ks: tuple[int],
        mean: torch.Tensor | None = None, 
        std: torch.Tensor | None = None
    ) -> None:
        
        self.ks = ks

        df = pd.read_csv(
            data_path,
            sep='\t'
        )

        self.kmer_counter = KmerCounter(
            ks=ks,
            stride=1,
            mode='soft',
            rna_alphabet=False,
            torch_output=True
        )

        self.kmers = [
            self.kmer_counter(seq)
            for seq in tqdm(
                df['seq'],
                desc='Calculating kmers...'
            )
        ]
        self.kmers = torch.stack(self.kmers).float()

        if mean is None:
            mean = self.kmers.mean(dim=0, keepdim=True)
        self.mean = mean
        
        if std is None:
            std = self.kmers.std(dim=0, keepdim=True)
        self.std = std

        self.kmers = (self.kmers - mean) / std

        self.labels = list(df['label'])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return {
            'seq': self.kmers[idx],
            'label': self.labels[idx]
        }
    
    def __len__(self) -> int:
        return len(self.kmers)
    

class KmerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds,
        val_ds,
        test_ds,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    

def create_datamodule(
    data_folder: str | Path,
    ks: tuple[int],
    batch_size: int,
    num_workers: int = 4
) -> tuple[DataLoader]:
    """
    Creates dataloaders and uses statistics from train fold
    for the normalization.
    
    :param data_folder: Description
    :type data_folder: str | Path
    :param ks: Description
    :type ks: tuple[int]
    :param batch_size: Description
    :type batch_size: int
    :return: Description
    :rtype: tuple
    """

    train_dataset = KmerDataset(
        data_path=data_folder / 'train.tsv',
        ks=ks
    )

    val_dataset = KmerDataset(
        data_path=data_folder / 'val.tsv',
        ks=ks,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    test_dataset = KmerDataset(
        data_path=data_folder / 'test.tsv',
        ks=ks,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    data_module = KmerDataModule(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return data_module
