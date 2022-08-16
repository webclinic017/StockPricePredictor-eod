import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union


class SplitData(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.split_ratio = None
        self.window_size = None
        self.dates = None
        self.debug = None

    def fit(self, split_ratio: float, window_size: int, dates: List, debug: bool):
        """
        """

        self.split_ratio = split_ratio
        self.window_size = window_size
        self.dates = dates
        self.debug = debug
        return self

    def transform(self, df):
        """
        """
        df_ = df.copy()

        # Make Calculations
        ttl_windows = len(df_)/self.window_size
        trainsubset = round(ttl_windows*self.split_ratio, 0)
        ttl_xtrain = trainsubset
        ttl_xval = +ttl_windows-trainsubset

        xtrain_split = trainsubset * self.window_size

        if self.debug == True:
            print("df:  ", len(df_))
            print("total windows in dataset: ", ttl_windows)
            print("\ntotal windows of {0}% train set: {1} ".format(
                self.split_ratio*100, trainsubset))
            print("total windows of {0}% valid set: {1} ".format(
                round((1 - self.split_ratio)*100), ttl_xval))

        # Do a split
        split = int(xtrain_split)
        time_train = self.dates[:split]
        x_train = df_[:split]
        time_valid = self.dates[split:]
        x_valid = df_[split:]

        # Print stuffs
        print("\nx_train window", len(x_train)/self.window_size)
        print("x_valid window", len(x_valid)/self.window_size)

        return x_train, x_valid
