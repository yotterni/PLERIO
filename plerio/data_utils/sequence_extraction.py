from Bio import SeqIO
from operator import itemgetter
from tqdm.auto import tqdm

import pandas as pd


class SequenceExtractor:
    """
    A class that stores parsed genome fasta file and extracts sequences
    given a bed-like `pd.DataFrame`. Note that compatibility between genome
    and coordinates is up to user and there is no checks implemented for it.
    The motivation for the `SequenceExtractor` is to open the genome
    dictionary only once and wrap it with this it.
    """
    def __init__(self, genome_file_path: str) -> None:
        """
        Parses a genome fasta file to the dictionary from headers
        to sequences. Possible chromosome names are `chr`-styled
        or `NC`-styled, others will lead to UB.
        :param genome_file_path: fasta file to be parsed.
        """
        genome_dict = SeqIO.to_dict(SeqIO.parse(genome_file_path, "fasta"))

        if not list(genome_dict.keys())[0].startswith('chr'):
            chromosome_names = ([f'chr{i}' for i in range(1, 23)] +
                                ['chrX', 'chrY', 'chrM'])
            name_pointer = 0
            for key in list(genome_dict.keys()):
                seq = genome_dict.pop(key)
                if 'NC' in key:
                    genome_dict[chromosome_names[name_pointer]] = seq
                    name_pointer += 1

        self.genome_dict = genome_dict
        self.getter = itemgetter(*('chr', 'start', 'end', 'strand'))

    def process_multiple_dataframes(
        self,
        dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Calls `self.__call__` for each dataframe in `dataframes`.
        :param dataframes: dictionary from dataframe name to dataframe.
        :return: dictionary from dataframe name to processed dataframe.
        """
        processed_dataframes = {}
        for dataframe_name, dataframe in tqdm(
            dataframes.items(),
            desc='Extracting sequences...'
        ):
            processed_dataframes[dataframe_name] = self.__call__(dataframe)
        return processed_dataframes

    def __call__(
        self, 
        df: pd.DataFrame, 
        show_tqdm: bool = False
    ) -> pd.DataFrame:
        """
        Adds to the coordinate bed-like dataframe `seq` column
        with corresponding sequences. Note that for each row this method
        will perform `seq[start:end + 1]`, where seq is the `seq` is the
        sequence of the corresponding chromosome.
        :param df: bed-like dataframe with sequence coordinates.
        :param show_tqdm: whether to show progress bars or not.
        :return: the same dataframe, but with additional `seq` column.
        """
        result_rows = []
        if show_tqdm:
            iterator = tqdm(
                df.iterrows(), 
                total=df.shape[0],
                # desc='Extracting sequences...'
            )
        else:
            iterator = df.iterrows()
        for idx, row in iterator:
            chrom, start, end, strand = self.getter(row)
            if chrom in self.genome_dict:
                seq = self.genome_dict[chrom][start:end + 1].seq.upper()
            else:
                continue

            assert strand in {'+', '-'}, 'Invalid strand!'
            if strand == '-':
                seq = seq.reverse_complement()
            # TODO: investigate the bug, when without str() 
            #  string becomes 200-letter tuple during tuple

            row['seq'] = str(seq)
            result_rows.append(row)
        return pd.DataFrame(result_rows)
