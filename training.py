import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
import numpy as np


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
            print("Dataframe shape:  ", df.shape)
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

        # Save extreme values
        x_train_extremes = x_train.iloc[:, 7:].copy()
        x_valid_extremes = x_valid.iloc[:, 7:].copy()

        # Remove extreme values
        x_valid = x_valid.iloc[:, :7].copy()
        x_train = x_train.iloc[:, :7].copy()

        print("--------> SplitData completed\n")
        return x_train, x_valid, x_train_extremes, x_valid_extremes


class GetTensoredDataset(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.window_size = None
        self.batch_size = None
        self.debug = None

    def windowed_dataset(self, series, window_size, batch_size):
        """
        Get windowed train dataset based on inputs
        """

        # This code must be trigered only when using Conv1D layer as input
        #series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=window_size, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        #ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[-1:, 1]))

        return ds.batch(batch_size).prefetch(1)

    def window_validation(self, x_valid, window_size, batch_valid):
        """
        create tensors of validation dataset
        """

        # Initialize data items
        counter = 0
        win = []
        ttl = []
        labels = []
        features = []
        temp_labels = []
        window = window_size
        x_valid_np = x_valid.to_numpy()

        # Loop row by row and skip each window in dataset (due to normalization that was done before)
        for item in range(0, len(x_valid), window_size):

            # Validation to not exceed the end of xvalid
            if item + window > len(x_valid):
                print(item + window)
                break

            # Get features of window
            while counter != window:
                if (counter+item) <= len(x_valid):
                    win.append(x_valid_np[counter+item])
                counter += 1

            if counter == window:
                counter = 0

            if counter+item > len(x_valid)-1:
                break

            # Get labels
            temp_labels.append(x_valid_np[item+window_size-1][1])

            labels.append(temp_labels)
            features.append(win[:-1])

            win = []
            temp_labels = []

        features = np.expand_dims(features, axis=-1)
        features = np.squeeze(features)

        val_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        val_dataset = val_dataset.batch(batch_valid)

        # rint("\nDone")

        return val_dataset, labels

    def fit(self, window_size: int, batch_size: int, train: bool, debug: bool):
        """
        """

        self.window_size = window_size
        self.batch_size = batch_size
        self.train = train
        self.debug = debug
        return self

    def transform(self, df: pd.DataFrame):
        """
        """
        labels = []
        if self.train == True:
            tensors = self.windowed_dataset(
                df, self.window_size, self.batch_size)
        else:

            tensors, labels = self.window_validation(
                df, self.window_size, self.batch_size)

        if self.debug == True:
            for batch in tensors:
                print(batch)
                break
        labels = np.squeeze(labels)

        print("--------> GetTensoredDataset completed\n")
        return tensors, labels
