import random
random.seed(42)

from tqdm.auto import tqdm

import pandas as pd
from pybedtools import BedTool


class NegativesCreator:
    """
    A class for negatives generation from the regions that are not
    intersected with given positives.
    """
    def __init__(self, genes_set_file_path: str,
                 window_size: int = 200, stride: int = 200) -> None:
        """
        :param genes_set_file_path: path to the bed file with genes set
        :param window_size: peaks size. To avoid overfitting on the
        window size, you should use the same window size you used for
        the positive peaks.
        :param stride: stride for the sliding window.
        """
        self.genes_set_file_path = genes_set_file_path
        self.number_of_positives = None
        self.negative_regions = pd.DataFrame()
        self.window_size = window_size
        self.stride = stride

    def process_multiple_dataframes(
        self,
        dataframes: dict[str, pd.DataFrame],
        negatives_fraction: float,
    ) -> dict[str, pd.DataFrame]:
        """
        Calls `self.process_single_dataframe` for each dataframe
        in `dataframes`.
        :param dataframes: dict from df name to df and 
        desired proportion of negatives to keep.
        :param negatives_fraction: for each particular df name in
        name_to_dataframe, if we'll construct a joint dataframe from
        its positives and its generated negatives, the proportion of
        negatives in the resulting dataframe will be equal
        to `negatives_fraction`.
        :return: dict from df_name to corresponding negative peaks.
        """
        negative_peak_dataframes = {}
        for df_name in tqdm(
            dataframes,
            desc='Sampling negative peaks...'
        ):
            df = dataframes[df_name]
            negative_peak_dataframes[df_name] = (
                self.process_single_dataframe(df, negatives_fraction))
        return negative_peak_dataframes

    def process_single_dataframe(
        self, 
        df: pd.DataFrame,
        negatives_fraction: float = 0.5
    ) -> pd.DataFrame:
        """
        High-level method that runs subtract from df and
        after that generate negatives from the result.
        :param df: dataframe with positive peaks
        :param negatives_fraction: resulting fraction of negatives
        :return: dataframe of negative `self.window_size`-nt peaks.
        """
        self.subtract_(df)
        return self.generate_negatives_(negatives_fraction)

    def subtract_(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the regions that are not intersected with given positives.
        It is equivalent to `genes_set subtract df` in the bedtools notation.
        :param df: dataframe with positive peaks
        :return: dataframe of annotated regions that don't overlap with df
        """
        self.number_of_positives = len(df)
        bed_df = BedTool.from_dataframe(df)
        genes_set = BedTool(self.genes_set_file_path)
        negative_regions = genes_set.subtract(bed_df).to_dataframe()
        negative_regions.rename(columns={'chrom': 'chr'}, inplace=True)
        negative_regions = negative_regions[['chr', 'start', 'end', 'strand']]
        self.negative_regions = negative_regions
        return negative_regions

    def generate_negatives_(
        self, 
        negatives_fraction: float = 0.5,
        show_tqdm: bool = False
    ) -> pd.DataFrame:
        """
        Samples `self.window_size`-nt peaks from given regions traversing them
        from the start to the end with sliding window of `self.window_size`
        size and `self.stride` stride. Of course, in the most cases it will
        generate too much samples, so parameter `keep_proportion` allows
        to save or not to save each sample with `keep_proportion` probability.
        :param negatives_fraction: required fraction of generated negatives
        in the dataframe with positives. It is used to calculate
        the `keep_probability` that generated negative peak will be kept.
        :return: dataframe of negative `self.window_size`-nt peaks.
        """
        keep_probability = (
            self.convert_fraction_to_sampling_probability_(
                negatives_fraction
            )
        )
        chrs = []
        starts = []
        ends = []
        strands = []

        if show_tqdm:
            iterator = tqdm(
                self.negative_regions.iterrows(),
                total=len(self.negative_regions),
                # desc='Generating negatives...'
            )
        else:
            iterator = self.negative_regions.iterrows()

        for idx, row in iterator:
            for start in range(
                row['start'], 
                row['end'] - self.window_size + 1,
                self.stride
            ):
                if random.uniform(0, 1) <= keep_probability:
                    starts.append(start)
                    ends.append(start + self.window_size)
                    strands.append(row['strand'])
                    chrs.append(row['chr'])

        negative_peaks = pd.DataFrame(
            {
                'chr': chrs, 
                'start': starts, 
                'end': ends,
                'strand': strands, 
                'label': [0] * len(starts)
            }
        )
        # print(len(negative_peaks))
        return negative_peaks

    def convert_fraction_to_sampling_probability_(
        self, 
        neg_frac: float
    ) -> float:
        """
        Converts the needed final negatives proportion to the
        negatives sampling probability.
        :param neg_frac: fraction of negative peaks in the joint dataframe
        of positives and negatives. Note that if `neg_frac` will be too close
        to 1, it might be possible that the maximum possible number of
        negative peaks won't be enough to satisfy it. In this case, reconsider
        the negative fraction you need or shorten the stride. It is also
        essential that in any adequate situation you won't reach this limit.
        :return: sampling probability.
        """
        assert neg_frac < 1
        # print(self.number_of_positives)
        required_number_of_negatives = int(
            neg_frac * self.number_of_positives / (1 - neg_frac)
        )
        # print('neg required:', required_number_of_negatives)

        lths = self.negative_regions['end'] - self.negative_regions['start']
        maximum_negative_peak_amount = sum(
            (lths - self.window_size) // self.stride
        )

        sampling_probability = (
                required_number_of_negatives / maximum_negative_peak_amount)
        assert sampling_probability <= 1, 'Not enough negatives!'
        return sampling_probability
