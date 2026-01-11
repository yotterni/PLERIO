from PLERIO_v2.plerio.data_utils.negatives_generation import NegativesCreator
from PLERIO_v2.plerio.data_utils.peak_processing import PeakProcessor
from pickling_batches import DatasetPickler
from PLERIO_v2.plerio.data_utils.sequence_extraction import SequenceExtractor
from PLERIO_v2.plerio.data_utils.train_val_test_splitting import MultipleDomainSplitter

from pathlib import Path

import os
import pandas as pd

#TODO: code one more object of filtering by rna groups.


class SingleProteinDatasetCreator:
    def __init__(self) -> None:
        pass

    def __call__(self, path: str | Path,
                 negatives_fraction: float = 0.5) -> None:
        narrow_peak_colnames = ['chr', 'start', 'end', 'protname',
                                'darkness', 'strand', 'signal',
                                'pval', 'wtf1', 'wtf2']

        protein_names = sorted(os.listdir(path))
        dataframes = {protein_name: pd.read_csv(os.path.join(
            path, protein_name, f'{protein_name}.narrowPeak'),
                                                names=narrow_peak_colnames,
                                                sep='\t')
                      for protein_name in protein_names}

        # print(len(dataframes['AARS']))
        print('Opened all files')
        peak_processor = PeakProcessor(200, tuple(range(-15, 16, 5)))
        positive_dataframes = peak_processor.process_multiple_dataframes(dataframes)

        for df_name, df in positive_dataframes.items():
            positive_dataframes[df_name]['label'] = [1] * len(df)
        # print(len(positive_dataframes['AARS']))

        neg_creator = NegativesCreator('../hg38/genes_set5.bed',
                                       200, 200)
        negative_dataframes = (
            neg_creator.process_multiple_dataframes(
                positive_dataframes,
                negatives_fraction=negatives_fraction))
        # print(len(negative_dataframes['AARS']))

        # concatenation and shuffling
        dataframes = {protein_name: pd.concat(
            [positive_dataframes[protein_name],
             negative_dataframes[protein_name]], axis=0).sample(frac=1)
            for protein_name in positive_dataframes}

        # print('before splitting:', len(dataframes['AARS']))

        rna_train_set = {'chr1', 'chr3', 'chr4', 'chr6',
                         'chr7', 'chr9', 'chr11', 'chr12',
                         'chr13', 'chr14', 'chr15', 'chr16',
                         'chr18', 'chr20', 'chr21', 'chrX'}

        rna_val_set = {'chr5', 'chr8', 'chr17', 'chr22'}
        rna_test_set = {'chr2', 'chr10', 'chrY', 'chr19'}

        mds = MultipleDomainSplitter({'chr': rna_train_set},
                                     {'chr': rna_val_set},
                                     {'chr': rna_test_set})
        print('Split all dataframes')

        train_dataframes = {df_name: mds.create_fold(df, 'train')
                            for df_name, df in dataframes.items()}
        val_dataframes = {df_name: mds.create_fold(df, 'val')
                            for df_name, df in dataframes.items()}
        test_dataframes = {df_name: mds.create_fold(df, 'test')
                          for df_name, df in dataframes.items()}

        # TODO: fix the path to the updated one (praga/hg38/genes_set5.bed)
        seq_extr = SequenceExtractor('../hg38/hg38.fna')
        train_dataframes = seq_extr.process_multiple_dataframes(train_dataframes)
        val_dataframes = seq_extr.process_multiple_dataframes(val_dataframes)
        test_dataframes = seq_extr.process_multiple_dataframes(test_dataframes)
        print('Extracted all sequences')
        # print(train_dataframes['AARS']['label'].mean())

        data_pickler = DatasetPickler(path, 8192)
        data_pickler.process_multiple_dataframes('train', train_dataframes)
        data_pickler.process_multiple_dataframes('val', val_dataframes)
        data_pickler.process_multiple_dataframes('train', test_dataframes)
        print('Pickled all the data...')

if __name__ == '__main__':
    datacreator = SingleProteinDatasetCreator()
    for cell_type in ['K562', 'HepG2']:
        print(f'Processing cell type {cell_type}')
        datacreator(f'encode_database/single_prot_dbs/{cell_type}', 0.9)
