from stringprep import in_table_d2
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class PullData(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """Initialize class with two parameters
        """
        super().__init__()
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.interval = None

        self.form_window = None
        self.target_window = None
        self.progress = None

        self.timeperiod1 = None
        self.timeperiod2 = None
        self.timeperiod3 = None

    def fit(self, ticker: str, start_date: str, end_date: str, interval: str, progress: bool, form_window: int, target_window: int, timeperiod1: int, timeperiod2: int, timeperiod3: int):
        # Data pulling
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.progress = progress

        # Data processing
        self.form_window = form_window
        self.target_window = target_window

        # EMA time periods
        self.timeperiod1 = timeperiod1
        self.timeperiod2 = timeperiod2
        self.timeperiod3 = timeperiod3

        return self

    def transform(self):
        """

        """
        # Load Data
        stock = yf.download(self.ticker,
                            start=self.start_date,
                            end=self.end_date,
                            interval=self.interval,
                            progress=self.progress,
                            )
        dataframe_ = stock.copy()

        # Remove nan
        dataframe_ = dataframe_.dropna(axis=0)

        # Add Indicators
        dataframe_['EMA' + str(self.timeperiod1)] = talib.EMA(
            dataframe_['Close'], timeperiod=self.timeperiod1)
        dataframe_['EMA' + str(self.timeperiod2)] = talib.EMA(
            dataframe_['Close'], timeperiod=self.timeperiod2)
        dataframe_['EMA' + str(self.timeperiod3)] = talib.EMA(
            dataframe_['Close'], timeperiod=self.timeperiod3)

        try:
            dataframe_ = dataframe_.drop(
                labels=['Adj Close', 'Volume'], axis=1)
        except:
            pass

        dataframe_ = dataframe_.reset_index()
        dataframe_['Date'] = pd.to_datetime(
            dataframe_['Date'], format="%Y-%m-%d")

        final_df_w = pd.DataFrame()

        for row in range(len(dataframe_)):

            if row + self.form_window + self.target_window < len(dataframe_):

                temp_df = pd.DataFrame()
                temp_df = dataframe_.iloc[row:row+self.form_window, :].copy()

                temp_df2 = pd.DataFrame()
                temp_df2 = dataframe_.iloc[row + self.form_window: row +
                                           self.form_window + self.target_window, :].copy()

                maxv = np.max(temp_df2.iloc[:, 1:4].to_numpy())
                minv = np.min(temp_df2.iloc[:, 1:4].to_numpy())

                if maxv == np.nan:
                    print(temp_df2.iloc[:, 1:4])
                    break

                openv = temp_df2.iloc[0, 1]
                closev = temp_df2.iloc[3, 4]

                dicti = {'Open': [openv],
                         'High': [maxv],
                         'Low': [minv],
                         'Close': [closev],
                         'Date': "Month"}

                temp_df3 = pd.DataFrame(dicti)

                final_df_w = pd.concat([final_df_w, temp_df], axis=0)
                final_df_w = pd.concat([final_df_w, temp_df3], axis=0)

        # remove nans (EMA)
        final_df_w = final_df_w.fillna(method='bfill')

        return final_df_w
