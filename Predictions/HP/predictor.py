from datetime import datetime
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
import talib


class MakeSinglePrediction(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()

        self.model_name = None
        self.form_window = None
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.interval = None
        self.form_window = None
        self.progress = None
        self.condition = None
        self.timeperiod1 = None
        self.timeperiod2 = None
        self.timeperiod3 = None
        self.debug = None
        self.acceptance = None
        self.penalization = None
        self.budget = None
        self.entry_candle = None
        self.sentiment = None
        self.news_df = None
        self.sentiment_type = None
        self.sentiment_aggr = None

    # Customized loss function
    def sign_penalty(y_true, y_pred):
        penalty = 100.
        loss = tf.where(tf.less(y_true*y_pred, 0),
                        penalty * tf.square(y_true-y_pred),
                        tf.square(y_true - y_pred)
                        )

        return(tf.reduce_mean(loss, axis=-1))

    # Customized functions
    def norm_df(self, pred1, mover):
        df_temp = pd.DataFrame()
        try:
            pred1 = pred1.drop('Date', axis=1)
        except:
            pass
        if mover == 1:
            pred_tmp = pred1.iloc[:, :-mover]
        else:
            pred_tmp = pred1.iloc[:, :]

        pred_np = pred_tmp.to_numpy()
        maxv = np.max(pred_np)
        minv = np.min(pred_np)
        df_temp = (pred_tmp-minv)/(maxv-minv)

        if mover == 1:
            news = pred1.iloc[:, -1]
            df_temp = pd.concat([df_temp, news], axis=1)

        return df_temp, maxv, minv

    def revert_df(self, df, maxv, minv, mover):
        df_temp = pd.DataFrame()

        if mover == 1:
            df_temp = (df.iloc[:, :-mover]*(maxv-minv))+minv
        else:
            df_temp = (df.iloc[:, :]*(maxv-minv))+minv

        return df_temp

    def revert_prediction(self, value, maxv, minv):

        return (value * (maxv-minv))+minv

    def MakePred(self, series, model):
        pr = series.to_numpy()
        series2 = np.array([[pr]])
        pred = tf.data.Dataset.from_tensor_slices(series2)
        prediction = model.predict(pred)

        return prediction

    def Profit_calculation(self, budget, entry, prediction):
        qty = round(budget / entry, 0)
        return (prediction - entry) * qty

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

        news_df['Week'] = news_df['Date'].apply(lambda x: x.week)
        news_df_agg = news_df.groupby('Week')[sentiment_type].mean()
        df_temp['Week'] = 0
        df_temp['Date'] = df_temp['Date'].astype('datetime64[ns]')
        df_temp['Date'] = [x.strftime("%Y-%m-%d") for x in df_temp['Date']]
        df_temp['Date'] = df_temp['Date'].astype('datetime64[ns]')
        df_temp['Week'] = df_temp['Date'].apply(lambda x: x.week)
        df_ttl = df_temp.merge(news_df_agg, on='Week', how='left')
        df_ttl = df_ttl.drop('Week', axis=1)

        return df_ttl

    def fit(self, model_name: str, form_window: int, ticker: str, start_date: str, end_date: str, interval: str,
            progress: bool, condition: bool, timeperiod1: int, timeperiod2: int, timeperiod3: int, debug: bool, budget: int,
            penalization: int, acceptance: int, entry_candle: str, sentiment: bool, news_df: pd.DataFrame, sentiment_type: str, sentiment_aggr: str):
        """
        """

        self.model_name = model_name

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.form_window = form_window

        self.progress = progress
        self.condition = condition
        self.timeperiod1 = timeperiod1
        self.timeperiod2 = timeperiod2
        self.timeperiod3 = timeperiod3
        self.debug = debug
        self.form_window = form_window

        self.budget = budget
        self.penalization = penalization
        self.acceptance = acceptance

        self.entry_candle = entry_candle
        self.sentiment = sentiment
        self.sentiment_type = sentiment_type
        self.news_df = news_df
        self.sentiment_aggr = sentiment_aggr
        stock = yf.download(self.ticker,
                            start=self.start_date,
                            end=self.end_date,
                            interval=self.interval,
                            progress=self.progress,
                            )
        stock = stock.dropna(axis=0)
        stock = stock.drop(labels=['Adj Close', 'Volume'], axis=1)

        stock[f'EMA{self.timeperiod1}'] = talib.EMA(
            stock['Close'], timeperiod=self.timeperiod1)
        stock[f'EMA{self.timeperiod2}'] = talib.EMA(
            stock['Close'], timeperiod=self.timeperiod2)
        stock[f'EMA{self.timeperiod3}'] = talib.EMA(
            stock['Close'], timeperiod=self.timeperiod3)

        # Reset index
        stock = stock.reset_index()

        # Get final dataframe
        # must be added 1 due to wrongly received data via yahoo api
        trading_formation = stock.tail(self.form_window+1)

        ###################
        if self.sentiment == True:
            trading_formation = self.AddSentimentAnalysis(
                trading_formation, self.news_df, self.sentiment_type)
            trading_formation.fillna(0, inplace=True)
        ###################

        if self.debug == True:
            print("Last Close: ", trading_formation.iloc[4, 4])
            print("Last Open: ", trading_formation.iloc[4, 1])
            print("Last High: ", trading_formation.iloc[4, 2])
            print("First open: ", trading_formation.iloc[0, 1])
            print(f"Last EMA{self.timeperiod2}: ",
                  trading_formation.iloc[4, 5])
            print(f"Last EMA{self.timeperiod3}: ",
                  trading_formation.iloc[4, 6])

        # Apply condition if needed
        if (self.condition == True):
            if (self.condition == True):  # add conditions
                entry = trading_formation.iloc[len(trading_formation)-1, 4]
                print("\nTrading condition passed, you can make prediction")
                print("\nEntry price: ", round(entry, 4))
            else:
                print("condition NOT passed, do NOT trade")

        return self, trading_formation

    def transform(self, df: pd.DataFrame):
        """
        """
        mover = 0
        if self.sentiment == True:
            mover = 1

        print("\nTicker: ", self.ticker)
        trading_formation = df.copy()
        tf.keras.losses.sign_penalty = self.sign_penalty
        model = tf.keras.models.load_model(self.model_name, custom_objects={
                                           'sign_penalty': self.sign_penalty})

        EntryPriceRow = 0

        def GetEntryPriceColl(candle):
            if candle == 'Current High':
                EntryPriceColl = 2
                EntryPriceRow = 0
            if candle == 'Current Open':
                EntryPriceColl = 1
                EntryPriceRow = 0
            if candle == 'Current Close':
                EntryPriceColl = 4
                EntryPriceRow = 0
            return EntryPriceColl, EntryPriceRow

        # print()
        entry_coll, entry_row = GetEntryPriceColl(self.entry_candle)

        def Predict(pred):
            df_temp1, maxv, minv = self.norm_df(pred, mover)
            pr = self.MakePred(df_temp1, model)
            prediction = self.revert_prediction(pr, maxv, minv)
            prediction = np.squeeze(prediction)
            return prediction

        entry = trading_formation.iloc[len(trading_formation)-1, entry_coll]

        pred = Predict(trading_formation)

        ppred = round(pred-self.penalization, 5)
        profit_pen = self.Profit_calculation(self.budget, entry, ppred)

        print(f'\nEntry candle ({self.entry_candle})')
        if np.max([pred, ppred])-entry > 0:
            print("\nBudget: ", self.budget)
            print("\nEntry price: ", round(entry, 2))
            print("Prediction: ", round(ppred, 2))
            print("Expected Market move: ", round(ppred - entry, 2))
            print("Expected Profit: ", round(profit_pen, 2))

        else:
            print("\nPrediction is NOT profitable")
            print("\nEntry price: ", round(entry, 2))
            print("Max Prediction: ", round(np.max([pred, ppred]), 2))
