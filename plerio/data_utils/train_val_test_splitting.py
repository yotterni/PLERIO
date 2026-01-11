from functools import reduce

import pandas as pd


class MultipleDomainSplitter:
    """
    A class to perform train-val-test splitting for data according to the
    given definitions of each fold. It is useful while dealing with
    dataset that can be divided into different domains by one or by each of
    many features. Each domain can be defined as a column and set of column
    values that corresponds to it, so user can construct a dictionary
    from column name to set of column values (each one will define a domain)
    and, meaning the intersection of those domains as a fold criteria,
    pass such a list to the splitter.
    """
    def __init__(
        self,
        train_criteria: dict[str, set],
        val_criteria: dict[str, set],
        test_criteria: dict[str, set]
    ) -> None:
        """
        :param train_criteria: train fold definition in terms of domains
        :param val_criteria: val fold definition in terms of domains
        :param test_criteria: test fold definition in terms of domains
        """
        self.train_criteria = train_criteria
        self.val_criteria = val_criteria
        self.test_criteria = test_criteria

    def __call__(
        self, 
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Just calls `self.__call__` for each fold and returns the result.
        :param df: dataframe that is to be split.
        :return: train, val, and test dataframes
        """
        train_df = self.create_fold(df, 'train')
        val_df = self.create_fold(df, 'val')
        test_df = self.create_fold(df, 'test')
        return train_df, val_df, test_df

    def create_fold(
        self, 
        df: pd.DataFrame, 
        fold: str
    ) -> pd.DataFrame:
        """
        Creates a fold as an intersection of domains intersecting
        the bitmasks for each domain.
        :param df: dataframe containing all columns that are used
        for domain definition.
        :param fold: required fold: train, val or test.
        :return: dataframe corresponding to the needed data fold.
        """
        if fold == 'train':
            criteria = self.train_criteria
        elif fold == 'val':
            criteria = self.val_criteria
        elif fold == 'test':
            criteria = self.test_criteria
        else:
            raise ValueError(f'Invalid fold value: {fold}')
        # print(len(df))

        masks = [
            df[colname].isin(acceptable_values)
            for colname, acceptable_values in criteria.items()
        ]
        # print([sum(mask) for mask in masks])
        mask = reduce(lambda x, y: x & y, masks)
        # print(sum(mask))
        return df[mask]
