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
        self.export_excel = None
        self.excel_path = None
        self.sentiment = None
        self.test_set = None
        self.validation_set = None

    def fit(self, split_ratio: float, window_size: int, dates: List, debug: bool, 
            export_excel: bool, excel_path: str, sentiment: bool, 
            validation_set: float, test_set: float, shuffle:bool):
        """
        """

        self.split_ratio = split_ratio
        self.window_size = window_size
        self.dates = dates
        self.debug = debug
        self.export_excel = export_excel
        self.excel_path = excel_path
        self.sentiment = sentiment
        self.validation_set = validation_set
        self.test_set = test_set
        self.shuffle = shuffle
        return self

    def transform(self, df):
        """
        """
        mover = 0
        if self.sentiment == True:
            mover = 1

        df_ = df.copy()

        # Validation
        if self.split_ratio + self.validation_set + self.test_set != 1:
            raise Exception("Error with ratio splits.")

 
        # Make Calculations
        ttl_windows = len(df_)/self.window_size

        trainsubset = round(ttl_windows*self.split_ratio, 0)
        xtrain_split = trainsubset * self.window_size
        
        testsubset = round(ttl_windows*self.test_set, 0)
        test_split = testsubset * self.window_size
        
        valsubset = round(ttl_windows * self.validation_set, 0)
        val_split = valsubset * self.window_size

        ttl_xval = + ttl_windows-trainsubset
        # get number for splitting

        # if self.debug == True:
        #     print("Dataframe shape:  ", df.shape)
        #     print("total windows in dataset: ", ttl_windows)
        #     print("\ntotal windows of {0}% train set: {1} ".format(
        #         self.split_ratio*100, trainsubset))
        #     print("total windows of {0}% valid set: {1} ".format(
        #         round((1 - self.split_ratio)*100), ttl_xval))

        # Do a split
        train_split = int(xtrain_split)
        val_split = int(val_split)
        test_split = int(test_split)

        print("DF Shape: ", df_.shape)
        print("train_split split: ", train_split)
        print("validation split: ", val_split)

        if self.shuffle==False:
            print("test split: ", val_split)

        #print("total validation windows: ", ttl_xval)

        if self.shuffle==True:
            #remove test set from dataframe, so that test set is not included in training
            time_total = self.dates[:-test_split].copy()
            df_total = df_[:-test_split].copy()
            #validation split
            time_valid = time_total[-val_split:]
            x_valid = df_total[-val_split:]
            # train split
            time_train = self.dates[:-val_split]
            x_train = df_total[:-val_split]
            # Format dates
            start_date_train = time_train.values[0]
            start_date_train = start_date_train.strftime('%Y-%m-%d')
            end_date_train = time_train.values[len(time_train)-2]
            end_date_train = end_date_train.strftime('%Y-%m-%d')
            # Format valid
            start_date_valid = time_valid.values[0]
            start_date_valid = start_date_valid.strftime('%Y-%m-%d')
            end_date_valid = time_valid.values[len(time_valid)-2]
            end_date_valid = end_date_valid.strftime('%Y-%m-%d')
        else:
            # test split
            time_test = self.dates[-test_split:]
            x_test = df_[-test_split:]
            #validation split
            time_valid = self.dates[-test_split-val_split:-test_split]
            x_valid = df_[-test_split-val_split:-test_split]
            # train split
            time_train = self.dates[:-test_split-val_split]
            x_train = df_[:-test_split-val_split]
            # Format dates
            start_date_train = time_train.values[0]
            start_date_train = start_date_train.strftime('%Y-%m-%d')
            end_date_train = time_train.values[len(time_train)-2]
            end_date_train = end_date_train.strftime('%Y-%m-%d')
            # Format valid
            start_date_valid = time_valid.values[0]
            start_date_valid = start_date_valid.strftime('%Y-%m-%d')
            end_date_valid = time_valid.values[len(time_valid)-2]
            end_date_valid = end_date_valid.strftime('%Y-%m-%d')
            # Format test
            start_date_test = time_test.values[0]
            start_date_test = start_date_test.strftime('%Y-%m-%d')
            end_date_test = time_test.values[len(time_test)-2]
            end_date_test = end_date_test.strftime('%Y-%m-%d')

        # Print stuffs
        print(f"\nSplit train ratio: {round(self.split_ratio*100)} %")
        print(f"Split validation ratio: {round(self.validation_set*100)} %")

        if self.shuffle==False:
            print(f"Split test ratio: {round(self.test_set*100)} %")
        print(
            f"\ntrain period: {start_date_train} - {end_date_train}")
        print(
            f"valid period: {start_date_valid} - {end_date_valid}")
        if self.shuffle==False:
            print(
                f"test period: {start_date_test} - {end_date_test}")
        if self.shuffle==True:
            print("\nTotal Windows (no test set): ",len(df_total)/self.window_size )
        else:
            print("\nTotal Windows: ", ttl_windows)

        print("x_train windows: ", len(x_train)/self.window_size)
        print("x_valid windows: ", len(x_valid)/self.window_size)

        if self.shuffle==False:
            print("x_test windows: ", len(x_test)/self.window_size)

        # Save extreme values
        x_train_extremes = x_train.iloc[:, 7+mover:].copy()
        x_valid_extremes = x_valid.iloc[:, 7+mover:].copy()

        # Remove extreme values
        x_valid = x_valid.iloc[:, :7+mover].copy()
        x_train = x_train.iloc[:, :7+mover].copy()

        if self.shuffle==False:
            x_test_extremes = x_test.iloc[:, 7+mover:].copy()
            x_test = x_test.iloc[:, :7+mover].copy()
        else:
            x_test_extremes = None
            x_test = None
            time_test = None

        if self.export_excel == True:
            x_valid.to_excel(f'{self.excel_path}/x_valid_dataset.xlsx')
            x_train.to_excel(f'{self.excel_path}/x_train_dataset.xlsx')
            #x_test.to_excel(f'{self.excel_path}/x_test_dataset.xlsx')

        
        print("--------> SplitData completed\n")
        return x_train, x_valid, x_test, x_train_extremes, x_valid_extremes, x_test_extremes, time_test


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
