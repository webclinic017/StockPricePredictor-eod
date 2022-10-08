import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import yfinance as yf
import talib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from datetime import datetime
from datetime import timedelta
import requests
import eod
from eod import EodHistoricalData
warnings.filterwarnings("ignore")

tf.random.set_seed(7788)
np.random.seed(7788)


class PullData(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.interval = None
        self.form_window = None
        self.target_window = None
        self.progress = None
        self.condition = None
        self.timeperiod1 = None
        self.timeperiod2 = None
        self.timeperiod3 = None
        self.export_excel = None
        self.excel_path = None

        self.listed_conditions = None

        self.sentiment = None
        self.news_df = None
        self.sentiment_type = None
        self.sentiment_aggr = None
        self.chart_period = None
        self.eod_API = None
        self.eod_key = None

    def fit(self, ticker: str, start_date: str, end_date: str, interval: str, progress: bool, condition: bool, form_window: int,
            target_window: int, timeperiod1: int, timeperiod2: int, timeperiod3: int, export_excel: bool, excel_path: str, listed_conditions: str,
            sentiment: bool, sentiment_type: str, news_df: pd.DataFrame, sentiment_aggr: str, chart_period: str, eod_API: bool, eod_key: str):

        # Data pulling
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.progress = progress
        self.condition = condition

        self.listed_conditions = listed_conditions
        # Data processing
        self.form_window = form_window
        self.target_window = target_window

        # EMA time periods
        self.timeperiod1 = timeperiod1
        self.timeperiod2 = timeperiod2
        self.timeperiod3 = timeperiod3

        self.export_excel = export_excel
        self.excel_path = excel_path

        self.sentiment = sentiment
        self.sentiment_type = sentiment_type
        self.news_df = news_df
        self.sentiment_aggr = sentiment_aggr
        self.chart_period = chart_period

        self.eod_key = eod_key
        self.eod_API = eod_API
        return self

    def AddSentimentAnalysis(self, df_temp, news_df, sentiment_type):
        """_summary_
        """
        news_df['Date'] = news_df['Date'].astype('datetime64[ns]')

        if self.sentiment_aggr == "mean":
            news_df_agg = news_df.groupby('Date')[sentiment_type].mean()
        if self.sentiment_aggr == "max":
            news_df_agg = news_df.groupby('Date')[sentiment_type].max()
        if self.sentiment_aggr == "min":
            news_df_agg = news_df.groupby('Date')[sentiment_type].min()
        if self.sentiment_aggr == "median":
            news_df_agg = news_df.groupby('Date')[sentiment_type].median()

        news_df_agg.to_excel('./Excel reports/adjusted_df_news.xlsx')

        df_temp['Date'] = df_temp['Date'].astype('datetime64[ns]')
        df_temp['Date'] = [x.strftime("%Y-%m-%d") for x in df_temp['Date']]
        df_temp['Date'] = df_temp['Date'].astype('datetime64[ns]')

        df_ttl = df_temp.merge(news_df_agg, on='Date', how='left')
        df_ttl[sentiment_type] = df_ttl[sentiment_type].fillna(0)
        return df_ttl

    def transform(self):
        """

        """
        mover = 0
        if self.sentiment == True:
            mover = -1

        # Load Data via different API calls
        if self.eod_API == False:
            stock = yf.download(self.ticker,
                                start=self.start_date,
                                end=self.end_date,
                                interval=self.interval,
                                progress=self.progress,
                                )

        else:

            client = EodHistoricalData(self.eod_key)
            stock = pd.DataFrame(client.get_prices_eod(
                self.ticker, period=self.interval, from_=self.start_date))
            print(stock.head(5))
            stock.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
                         'close': 'Close', 'adjusted_close': 'Adj Close', 'volume': 'Volume'}, inplace=True)
            stock.set_index('Date', inplace=True)

        dataframe_ = stock.copy()

        # Remove nan
        dataframe_ = dataframe_.dropna(axis=0)

        print("initial shape: ", stock.shape)

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

        ###################
        def AdjustDate(df):

            df['Date'] = df['Date'].astype('datetime64[ns]')
            df['Date'] = [x.strftime("%Y-%m-%d") for x in df['Date']]

            for row in range(df.shape[0]):
                #date = df.iloc[row, 0]
                df['Date'] = df['Date'].astype('datetime64[ns]')
                delta = df.iloc[row, 0].weekday()
                df.iloc[row, 0] = df.iloc[row, 0] - timedelta(days=delta)
            return df

        if self.sentiment == True:
            self.news_df.to_excel("./Excel reports/initial_news.xlsx")

            if self.chart_period == "1wk":
                self.news_df = AdjustDate(self.news_df)

            # self.news_df.to_excel("adjusted_news.xlsx")
            dataframe_ = self.AddSentimentAnalysis(
                dataframe_, self.news_df, self.sentiment_type)

        ###################
        final_df_w = pd.DataFrame()

        for row in range(len(dataframe_)):

            if row + self.form_window + self.target_window <= len(dataframe_):

                temp_df = pd.DataFrame()
                temp_df = dataframe_.iloc[row:row+self.form_window, :].copy()

                temp_df2 = pd.DataFrame()
                temp_df2 = dataframe_.iloc[row + self.form_window: row +
                                           self.form_window + self.target_window, :].copy()

                maxv = np.max(temp_df2.iloc[:, 1:4].to_numpy())
                minv = np.min(temp_df2.iloc[:, 1:4].to_numpy())
                # print(temp_df2)
                # break
                if maxv == np.nan:
                    #print(temp_df2.iloc[:, 1:4])
                    break

                openv = temp_df2.iloc[0, 1]
                closev = temp_df2.iloc[(self.target_window - 1), 4]

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
        final_df_w = final_df_w.fillna(method='ffill')

        # # Apply condition
        trades = 0
        final_df = pd.DataFrame()

        # Conditions possible to place in if decission

        # high of trades month is above previous high - we are entering at high of previous week - !!! Entry must be high of prev week
        condition0 = "final_df_w.iloc[row-1, 2] < final_df_w.iloc[row, 2]"
        # Last Close is lower than first indicator
        condition1 = "final_df_w.iloc[row-1, 4] < final_df_w.iloc[row-1, 5]"
        # first indicator is lower then second indicator
        condition2 = "final_df_w.iloc[row-1, 5] < final_df_w.iloc[row-1, 6]"
        # second indicator is lower then third indicator
        condition3 = "final_df_w.iloc[row-1, 6] < final_df_w.iloc[row-1, 7]"
        # Last close is above 1st indicator
        condition4 = "final_df_w.iloc[row-1, 4] > final_df_w.iloc[row-1, 5]"
        # first indicator is above second indicator
        condition5 = "final_df_w.iloc[row-1, 5] > final_df_w.iloc[row-1, 6]"
        # second indicator is above third indicator
        condition6 = "final_df_w.iloc[row-1, 6] > final_df_w.iloc[row-1, 7]"

        dicti = {'high_entry': condition0,
                 'condition1': condition1,
                 'condition2': condition2,
                 'condition3': condition3,
                 'condition4': condition4,
                 'condition5': condition5,
                 'condition6': condition6}
        # if self.condition == True:
        #     for item in self.listed_conditions:

        for row in range(self.form_window, len(final_df_w)):

            if final_df_w.iloc[row, 0] == "Month":

                if (self.condition == True and eval(condition0)):
                    temp_df = pd.DataFrame()

                    temp_df = final_df_w.iloc[row-self.form_window:row+1, :]

                    trades += 1

                    temp_df = final_df_w.iloc[row-self.form_window:row+1, :]

                    temp_df['trades'] = int(trades)

                    final_df = pd.concat([final_df, temp_df], axis=0)

                if self.condition == False:
                    temp_df = pd.DataFrame()

                    temp_df = final_df_w.iloc[row-self.form_window:row+1, :]

                    trades += 1

                    temp_df = final_df_w.iloc[row-self.form_window:row+1, :]

                    temp_df['trades'] = int(trades)

                    final_df = pd.concat([final_df, temp_df], axis=0)

        final_df_w = final_df.copy()

        if self.export_excel == True:
            final_df_w.to_excel(
                f'{self.excel_path}/{self.ticker}_windowed_dataset.xlsx')
            stock.to_excel(f'{self.excel_path}/{self.ticker}_raw_dataset.xlsx')

            print("Output shape: ", final_df_w.shape)
            print("--------> PullData completed\n")
        return final_df_w


class NormalizeData(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.window_size = None
        self.shuffle = None
        self.debug = None
        self.export_excel = None
        self.excel_path = None
        self.sentiment = None

    def fit(self, window_size: int, shuffle: bool, debug: bool, export_excel: bool, excel_path: str, sentiment: int):
        """
        """

        self.window_size = window_size
        self.shuffle = shuffle
        self.debug = debug
        self.export_excel = export_excel
        self.excel_path = excel_path
        self.sentiment = sentiment
        return self

    def transform(self, df):
        """
        """
        mover = ""
        if self.sentiment == True:
            mover = -1

        # Print stuffs
        print("Dataframe shape: ", df.shape)
        formations = int(df.shape[0]/self.window_size)
        print("Number of formations: ", formations)
        len_initial = df.shape[0]

        # Shuffle if True
        if self.shuffle == True:
            df_ = df.copy()
            # 3D reshaping
            temp = df_.values.reshape(-1, self.window_size, df_.shape[1])
            # shuffling
            sh = shuffle(temp)
            # return back to DF
            temp_df = sh.reshape(df_.shape[0], df_.shape[1])
            df = pd.DataFrame(temp_df, columns=df_.columns)

            # after reshuffling columns are object type, must be changed to float
            for coll in df.columns:
                if df[coll].dtype == 'object' and coll != 'Date' and coll != 'trades':
                    df[coll] = df[coll].astype('float64')

        if self.export_excel == True and self.shuffle == True:
            df.to_excel(f'{self.excel_path}/reshufled_dataset.xlsx')

        # Get separated Date and remove it from df
        if 'Date' in df.columns:
            Dates = df.iloc[:len_initial, 0]
            df = df.iloc[:df.shape[0], 1:]

        # Drop trades column from dataset
        if 'trades' in df.columns:
            df = df.drop('trades', axis=1)

        # Initialize dataitems
        counter = 0
        inc = 0
        Highs = []
        Lows = []
        df_norm = pd.DataFrame()

        # Loop through each window separately to normalize
        for row in range(0, len(df), self.window_size):
            counter += 1
            df_temp = pd.DataFrame()

            # Get maxv and minv of window
            while inc < df.shape[1]:
                # Get max High
                if row + inc < len(df):

                    Highs.append(df.iloc[row+inc][1])
                    Lows.append(df.iloc[row+inc][2])

                    inc += 1
                else:
                    break

            # reset inc
            inc = 0

            # Save Max and Min
            maxv = max(Highs)
            minv = min(Lows)

            # testing
            if self.sentiment == True:
                maxv = np.max(
                    df.iloc[row:row + self.window_size, :mover].to_numpy())
                minv = np.min(
                    df.iloc[row:row + self.window_size, :mover].to_numpy())
            else:
                maxv = np.max(
                    df.iloc[row:row + self.window_size, :].to_numpy())
                minv = np.min(
                    df.iloc[row:row + self.window_size, :].to_numpy())
            # print(maxv)
            # print(minv)
            # break
            # Reset
            Highs = []
            Lows = []

            # Print first 2 windows for checking
            if counter < 3 and self.debug == True:
                # Print data windowing
                print("\nWindow:" + str(counter) + "\n " +
                      str(df.iloc[row:row + self.window_size, :]))
                print("\nMax value is ", maxv)
                print("Min value is ", minv)
                # print("\n Normalized:\n " +
                #       str((df.iloc[row:row + self.window_size, :]-minv)/(maxv-minv)))

            # Merge normalized window to new dataframe
            if self.sentiment == True:
                df_temp = (df.iloc[row:row + self.window_size,
                                   :mover]-minv)/(maxv-minv)
            else:
                df_temp = (df.iloc[row:row + self.window_size,
                                   :]-minv)/(maxv-minv)
            df_temp['maxv'] = maxv
            df_temp['minv'] = minv

            if self.sentiment == True:
                temp_news = df.iloc[row:row + self.window_size, -1]
                # print(temp_news)
                df_temp = pd.concat([df_temp, temp_news], axis=1)
                df_temp = df_temp.iloc[:, [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]]
                # print(df_temp)
            df_norm = pd.concat([df_norm, df_temp], axis=0)

        # rearrange columns in df so that maxv and minv are at the end of df

        if self.export_excel == True:
            df_norm.to_excel(f'{self.excel_path}/normalized_dataset.xlsx')

        print("--------> NormalizeData completed\n")

        return df_norm, Dates


class ReverseNormalization(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.forecasts = None
        self.window_size = None
        self.labels = None
        self.debug = None
        self.x_valid = None
        self.x_valid_x = None
        self.sentiment = None
        self.sentiment_type = None

    def RevertNorm(self, df_final, window_size):
        """_summary_
        """
        mover = 0
        if self.sentiment == True:
            mover = -1
        # Initialize dataitems
        counter = 0
        inc = 0
        Highs = []
        Lows = []
        df_rev = pd.DataFrame()

        # Loop through each window separately to normalize
        for row in range(0, len(df_final), window_size):

            #print("\nCurrent row is: ",row)
            counter += 1
            df_temp = pd.DataFrame()

            # Get maxv and minv of window
            while inc < df_final.shape[1]:
                # Break for loop in case of excession
                if row + inc < len(df_final):

                    inc += 1
                else:
                    break

            # reset inc
            inc = 0

            maxv = np.squeeze(df_final.iloc[row, -2])  # .to_numpy()
            minv = np.squeeze(df_final.iloc[row, -1])  # .to_numpy()

            if self.debug == True:
                #########################
                # Debugging block
                # print("maxv: ",maxv)
                # print("minv: ",minv)
                # break
                # Print first 2 windows for checking
                if counter < 3:
                    # Print data windowing
                    print("\nWindow:" + str(counter) + "\n " +
                          str(df_final.iloc[row:row + window_size]))
                    print("\nMax value is ", maxv)
                    print("Min value is ", minv)
                    print(
                        "\n Reverted:\n " + str((df_final.iloc[row:row + window_size]*(maxv-minv))+minv))
                ##############################

            if self.sentiment == True:
                # get sentiment data out of df
                news_df = df_final.iloc[row:row + window_size, -5]

            ddf = df_final.iloc[row:row + window_size, :]

            try:
                ddf = ddf.drop(self.sentiment_type, axis=1)
            except:
                pass

            # Merge normalized window to new dataframe
            df_temp = (ddf*(maxv-minv))+minv
            # df_temp = (df_final.iloc[row:row +
            #            window_size, :]*(maxv-minv))+minv

            if self.sentiment == True:
                df_temp = pd.concat([df_temp, news_df], axis=1)

            df_rev = pd.concat([df_rev, df_temp], axis=0)

        # get final df, remove maxv and minv from df #df_rev = df_rev.iloc[:, :]
        df_rev = df_rev.drop(labels=['maxv', 'minv'], axis=1)

        return df_rev

    def fit(self, forecasts: List, labels: List, window_size: int, debug: bool, x_valid: pd.DataFrame, x_valid_x: pd.DataFrame, sentiment: bool, sentiment_type: str):
        """
        """

        self.forecasts = forecasts
        self.labels = labels
        self.window_size = window_size
        self.x_valid = x_valid
        self.x_valid_x = x_valid_x
        self.debug = debug
        self.sentiment = sentiment
        self.sentiment_type = sentiment_type
        return self

    def transform(self):
        """
        """
        # create two columns with labels and predictions
        y_prediction = []
        y_labels = []
        counter = 0
        #x_valid_new = pd.DataFrame()
        x_valid_ = self.x_valid

        for item in range(len(self.forecasts)):

            while counter < self.window_size-1:
                y_prediction.append(np.nan)
                y_labels.append(np.nan)
                counter += 1

                if counter == self.window_size-1:
                    y_prediction.append(self.forecasts[item])
                    y_labels.append(self.labels[item])
            counter = 0

        y_labels = np.squeeze(y_labels)
        dicti = {'labels': y_labels,
                 'prediction': y_prediction,
                 'In': [counter2 for counter2 in range(len(y_prediction))]
                 }

        prediction_df = pd.DataFrame(dicti)
        prediction_df = prediction_df.set_index('In')

        # add extreme values to be able to revert normalization
        self.x_valid_x["In"] = np.arange(len(self.x_valid))
        self.x_valid_x = self.x_valid_x.set_index('In')
        x_valid_['In'] = np.arange(len(self.x_valid))
        x_valid_ = x_valid_.set_index('In')

        # merge all to one
        df_valid_norm = pd.concat(
            [x_valid_, prediction_df, self.x_valid_x], axis=1)

        # Revert normalization
        df_rev = self.RevertNorm(df_valid_norm, self.window_size)

        print("--------> ReverseNormalization completed\n")
        return df_rev
