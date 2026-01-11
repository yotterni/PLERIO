from directory_checking import DirectoryChecker
from PLERIO_v2.plerio.data_utils.kmer_counting import KmerCounter

from tqdm.auto import tqdm

import os
import typing as tp

import numpy as np
import pandas as pd
import pickle
import torch


class DatasetPickler:
    """
    A class for transformation of `pd.DataFrame` with sequences, labels
    and user-defined additional columns into the folder of pickled batches
    consists of kmer sequence representations, labels and user-defined
    metadata.
    """
    def __init__(self, path: str, batch_size: int,
        additional_cols: tp.Iterable[str] | None = None,
        kmer_params: dict[str, tp.Any] = {
            'ks': (3, 5),
            'torch_output': True
        },
        keep_existing_dirs: bool = False,
        ) -> None:
        """
        Makes sure that the passed path exists (or recreates it) and then
        initializes the class.
        :param path: path to folder where folders with pickled batches
         will be stored.
        :param batch_size: size of the batch.
        :param additional_cols: values from this columns will be also kept
         while pickling the batches alongside with sequences and proteins.
        :param kmer_params: dictionary from parameter to its value for
         initializing the `KmerCounter`.
        :param keep_existing_dirs: whether to keep existing folders or not,
         parameter for initializing the `DirectoryChecker`.
        """
        self.dir_checker = DirectoryChecker(keep_existing=keep_existing_dirs)
        self.dir_checker.handle(path)

        self.path = path
        self.batch_size = batch_size
        self.kmer_counter = KmerCounter(**kmer_params)
        self.additional_cols = additional_cols

    def __call__(self, df_name: str, fold: str, df: pd.DataFrame) -> None:
        current_path = os.path.join(self.path, df_name, f'{fold}_dataset')
        self.dir_checker.handle(current_path)
        for idx in range(0, max(len(df), self.batch_size), self.batch_size):
            batch_df = df.iloc[idx: idx + self.batch_size, :]
            batch_representations = torch.Tensor(
                np.vstack([self.kmer_counter(seq)
                           for seq in batch_df['seq']]))
            batch_labels = torch.Tensor(np.array(batch_df['label']))

            additional_arrays_list = []
            if self.additional_cols is not None:
                for col in self.additional_cols:
                    additional_arrays_list.append(list(batch_df[col]))

            pickle_tuple = tuple([batch_representations,
                                  *additional_arrays_list,
                                  batch_labels])

            with open(os.path.join(current_path, f'{idx}.pkl'), 'wb') as file:
                pickle.dump(pickle_tuple, file)

    def process_multiple_dataframes(self, fold: str,
                                    dataframes: dict[str, pd.DataFrame]
                                    ) -> None:
        """
        Calls `self.__call__` for each dataframe in `dataframes`.
        Each dataframe name (dict key) will be used as a suffix for the path
        to the location of the pickled batches.
        :param fold: fold name: train, val or test.
        :param dataframes: dictionary from dataframe name to `pd.DataFrame`,
         each dataframe must contain at least `seq` and `label` columns.
        :return: `None`
        """
        for dataframe_name, dataframe in tqdm(
            dataframes.items(),
            desc='Pickling the data...'
        ):
            self.__call__(dataframe_name, fold, dataframe)
