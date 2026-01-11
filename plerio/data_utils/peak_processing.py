import pandas as pd


class PeakProcessor:
    """
    A class for peaks processing. It can enlarge or shorten them
    to the given length (`window_size`) and augment them using
    the left-right shift.
    """
    def __init__(
        self, 
        window_size: int = 200,
        augmentation_strides: tuple[int] = (-5, 0, 5)
    ) -> None:
        self.window_size = window_size
        self.augmentation_strides = augmentation_strides

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A high-level API for calling PeakProcessor's methods one
        by another. Peaks are:
        1) enlarged via `self.enlarge_peaks`
        2) augmented via `self.augmented_peaks`
        :param df: `pd.DataFrame` with 'start' and 'end' columns
        or any iterable that returns such dataframe on each
        iteration.
        :return: pd.DataFrame with all original columns, but
        consists of enlarged and then augmented peaks.
        """
        enlarged_df = self.enlarge_peaks(df)
        return self.augment_peaks(enlarged_df)

    def process_multiple_dataframes(
        self, 
        dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Calls `self.__call__` for each dataframe in `dataframes` and returns
        the mapping from original dataframe name to the processed dataframe.
        :param dataframes: dictionary from dataframe name to dataframe.
        :return: dictionary from dataframe names to processed dataframes.
        """
        processed_dataframes = {}
        for dataframe_name, dataframe in dataframes.items():
            processed_dataframes[dataframe_name] = self.__call__(dataframe)
        return processed_dataframes

    def enlarge_many_peak_dataframes(
        self, 
        dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Calls `self.enlarge_peaks` for each dataframe in `dataframes` and returns
        the mapping from original dataframe name to the processed dataframe.
        :param dataframes: dictionary from dataframe name to dataframe.
        :return: dictionary from dataframe names to processed dataframes.
        """
        processed_dataframes = {}
        for dataframe_name, dataframe in dataframes.items():
            processed_dataframes[dataframe_name] = (
                self.enlarge_peaks(dataframe))
        return processed_dataframes


    def enlarge_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Each peak is reduces to the length of `self.window_size`. If
        the peak is shorter, it will be enlarged, if the peak is longer,
        it will be clipped. Both operations mean symmetric operations
        with respect to the peak center.
        :param df: pd.DataFrame with 'start' and 'end' columns.
        :return: the same pd.DataFrame, but with modified 'start'
        and 'end' columns: each peak is reduces to have length
        of `window_size`.
        """
        rows = []
        for idx, row in df.iterrows():
            start, end = row['start'], row['end']
            peak_length = end - start
            margin = abs(self.window_size - peak_length)
            if peak_length > self.window_size:
                if margin % 2 == 0:
                    start += margin // 2
                    end -= margin // 2
                else:
                    start += margin // 2
                    start += 1
                    end -= margin // 2
            else:
                if margin % 2 == 0:
                    start -= margin // 2
                    end += margin // 2
                else:
                    start -= margin // 2
                    start -= 1
                    end += margin // 2
            row['start'] = start
            row['end'] = end
            rows.append(row)
        return pd.DataFrame(rows)

    def augment_peaks(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Create augmentations for each peak by shifting its borders
        to the left and to the right by the number of nucleotides declared
        in the `self.augmentation_strides` tuple.
        :param df: pd.DataFrame with 'start' and 'end' columns.
        :return: pd.DataFrame that is `len(self.augmentation_strides)` times
        longer than the original dataframe and contains peak augmentations.
        Note that original peaks will be presented if only
        `self.augmentation_strides` contains `0` stride.
        """
        # TODO: rewrite it in generator manner using `yield`
        # probably can be optimized by memory
        # if we won't store any coord and seq files at all)
        augmented_rows = []
        for idx, row in df.iterrows():
            start, end = row['start'], row['end']
            for stride in self.augmentation_strides:
                current_aug = row.copy()
                current_aug['start'] = start + stride
                current_aug['end'] = end + stride
                augmented_rows.append(current_aug)
        return pd.DataFrame(augmented_rows)
