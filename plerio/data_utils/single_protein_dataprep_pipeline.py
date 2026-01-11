import sys
sys.path.append('.')


from plerio.data_utils.negatives_generation import NegativesCreator
from plerio.data_utils.peak_processing import PeakProcessor
from plerio.data_utils.sequence_extraction import SequenceExtractor
from plerio.data_utils.train_val_test_splitting import MultipleDomainSplitter

from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import os


narrow_peak_colnames = [
    'chr', 
    'start', 
    'end', 
    'protname',
    'darkness', 
    'strand', 
    'signal',
    'pval', 
    '_1', 
    '_2'
]

def save_datasets(
    fold_frames: tuple[dict[str, pd.DataFrame], ],
    destination_folder: str | Path
) -> None:
    destination_folder = Path(destination_folder)

    assert len(fold_frames) == 3
    train_frames, val_frames, test_frames = fold_frames
    for protein_name in tqdm(
        train_frames,
        desc='Saving dataframes...'
    ):
        train_frames[protein_name][['seq', 'label']].to_csv(
            destination_folder / protein_name / 'train.tsv',
            sep='\t',
            index=False
        )

        val_frames[protein_name][['seq', 'label']].to_csv(
            destination_folder / protein_name / 'val.tsv',
            sep='\t',
            index=False
        )

        test_frames[protein_name][['seq', 'label']].to_csv(
            destination_folder / protein_name / 'test.tsv',
            sep='\t',
            index=False
        )

def create_datasets_for_cell_type(
    data_dir: str | Path,
    cell_type: str,
    genome_fasta_path: str | Path,
    genes_set_path: str  | Path,
    negatives_fraction: float = 0.5,

) -> None:
    data_dir = Path(data_dir)

    protein_names = sorted(os.listdir(data_dir / cell_type))
    dataframes = {
        protein_name: pd.read_csv(
            data_dir / cell_type / protein_name / f'{protein_name}.narrowPeak',
            names=narrow_peak_colnames,
            sep='\t'
        )
        for protein_name in protein_names
    }

    peak_processor = PeakProcessor(200, tuple(range(-15, 16, 5)))
    positive_dataframes = peak_processor.process_multiple_dataframes(
        dataframes
    )

    for df_name, df in tqdm(
        positive_dataframes.items(),
        desc='Aligning & augmenting positive peaks'
    ):
        positive_dataframes[df_name]['label'] = [1] * len(df)

    neg_creator = NegativesCreator(
        genes_set_path,
        window_size=200, 
        stride=200
    )
    negative_dataframes = (
        neg_creator.process_multiple_dataframes(
            positive_dataframes,
            negatives_fraction=negatives_fraction,
        )
    )

    dataframes = {
        protein_name: pd.concat(
            [
                positive_dataframes[protein_name],
                negative_dataframes[protein_name]
            ], axis=0
        ).sample(frac=1) # for shuffling
        for protein_name in positive_dataframes
    }

    rna_train_set = {'chr1', 'chr3', 'chr4', 'chr6',
                         'chr7', 'chr9', 'chr11', 'chr12',
                         'chr13', 'chr14', 'chr15', 'chr16',
                         'chr18', 'chr20', 'chr21', 'chrX'}

    rna_val_set = {'chr5', 'chr8', 'chr17', 'chr22'}
    rna_test_set = {'chr2', 'chr10', 'chrY', 'chr19'}

    mds = MultipleDomainSplitter({'chr': rna_train_set},
                                        {'chr': rna_val_set},
                                        {'chr': rna_test_set})

    train_dataframes = {
        df_name: mds.create_fold(df, 'train')
        for df_name, df in dataframes.items()
    }
    val_dataframes = {
        df_name: mds.create_fold(df, 'val')
        for df_name, df in dataframes.items()
    }
    test_dataframes = {
        df_name: mds.create_fold(df, 'test')
        for df_name, df in dataframes.items()
    }

    seq_extr = SequenceExtractor(genome_fasta_path)
    train_dataframes = seq_extr.process_multiple_dataframes(train_dataframes)
    val_dataframes = seq_extr.process_multiple_dataframes(val_dataframes)
    test_dataframes = seq_extr.process_multiple_dataframes(test_dataframes)

    save_datasets(
        (
            train_dataframes,
            val_dataframes,
            test_dataframes
        ),
        data_dir / cell_type
    )


if __name__ == '__main__':
    for cell_type in ['K562', 'HepG2']:
        print(f'Processing cell type {cell_type}')
        create_datasets_for_cell_type(
            data_dir='data',
            cell_type=cell_type,
            genome_fasta_path='../hg38/hg38.fna',
            genes_set_path='../hg38/genes_set5.bed',
            negatives_fraction=0.7
        )