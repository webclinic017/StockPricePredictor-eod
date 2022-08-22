from datetime import datetime
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
import talib


class GetFinalDataframe(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()
        self.dates = None
        self.x_valid = None

    def fit(self, dates: List, x_valid: pd.DataFrame):
        """
        """

        self.dates = dates
        self.x_valid = x_valid
        return self

    def transform(self, df: pd.DataFrame):
        """
        """
        df_ = df.copy()
        try:
            self.x_valid.drop('In', axis=1, inplace=True)
        except:
            pass

        # Get start of validation dataset
        val_start = len(self.dates)-self.x_valid.shape[0]
        Dates_val = self.dates[val_start:]
        dicti = {"Datetime": Dates_val,
                 'In': np.arange(len(Dates_val))
                 }
        dates_df = pd.DataFrame(dicti)
        df_ = df_.fillna("nn")
        dates_df = dates_df.set_index('In')
        merged_df = pd.concat([df_, dates_df], axis=1)

        # Remove Month from datetime column
        for row in range(len(merged_df)):
            if merged_df.iloc[row, 9] == "Month":
                merged_df.iloc[row, 9] = "2000-11-07 00:00:00"

        merged_df['Datetime'] = pd.to_datetime(merged_df['Datetime'])

        # Get datetime column to proper format
        for row in range(len(merged_df)):
            start = str(merged_df.iloc[row, -1])
            start_date = datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
            new_date = start_date.strftime('%Y-%m-%d')
            # print(new_date)
            merged_df.iloc[row, -1] = new_date
        # Change date to right value of Month rows

        for row in range(len(merged_df)):
            start = merged_df.iloc[row, -1]
            start = start.strftime('%Y-%m-%d')
            start_date = datetime.strptime(start, '%Y-%m-%d').date()
            merged_df.iloc[row, -1] = start_date + timedelta(days=0)

            if merged_df.iloc[row, -2] != "nn":
                start = merged_df.iloc[row-1, -1]
                start = start.strftime('%Y-%m-%d')
                start_date = datetime.strptime(start, '%Y-%m-%d').date()
                merged_df.iloc[row, -1] = start_date + timedelta(days=1)

        print("--------> GetFinalDataframe\n")
        return merged_df


class GetModelPerformance(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        entry candle: Current Open = Previous High = Previous Close

        """

        super().__init__()
        self.acceptance = None
        self.penalization = None
        self.entry_candle = None
        self.window_size = None
        self.budget = None
        self.export_excel = None
        self.excel_path = None
        self.excel_path = None

    def profit_calculation(self, difference, stock_price, budget):
        qty = round(budget/stock_price, 0)

        return round(difference * qty)

    def fit(self, acceptance: int, penalization: int, entry_candle: str, window_size: int, budget: int, export_excel: bool, excel_path: str):
        """
        """
        self.acceptance = acceptance
        self.penalization = penalization
        self.entry_candle = entry_candle
        self.window_size = window_size
        self.budget = budget
        self.export_excel = export_excel
        self.excel_path = excel_path

        return self

    def transform(self, df: pd.DataFrame):
        """
        """
        print(f"period: {df.iloc[0,9]} - {df.iloc[df.shape[0]-2,9]}")
        print("Formations: ", int(df.shape[0]/self.window_size))

        # Initialize data items
        df_ = df.copy()
        EntryPrice = 0
        ttl_profit = 0
        ttl_loss = 0
        profit = 0
        loss = 0
        temp_app = 0
        app_profit = 0
        counter = 0
        trade_counter = 0
        ttl_diff = 0
        exact_loss = 0
        loss_trades = 0
        profit_trades = 0
        trades_df = pd.DataFrame()
        exact_profit = 0
        ccc = 0

        #######################################################################
        for candle in range(0, len(df_), 1):
            # print(candle)
            temp_df = pd.DataFrame()
            difference_value = 0

            label = df_.iloc[candle, -3]
            predictions = df_.iloc[candle, -2]

            if label != "nn" and predictions != "nn":
                predictions = df_.iloc[candle, -2] - self.penalization

                prev_high = df_.iloc[candle-1, 1]
                current_high = df_.iloc[candle, 1]
                current_close = df_.iloc[candle, 3]
                current_open = df_.iloc[candle, 0]
                current_low = df_.iloc[candle, 2]
                prev_close = df_.iloc[candle-1, 3]

                def GetEntryPrice(candle):
                    if candle == 'Previous High':
                        EntryPrice = prev_high
                    if candle == 'Current Open':
                        EntryPrice = current_open
                    if candle == 'Previous Close':
                        EntryPrice = prev_close
                    return EntryPrice

                EntryPrice = GetEntryPrice(self.entry_candle)
        #         print("current high: ",current_high)
        #         print("previous high: ",prev_high)
        #         print("current open: ",current_open)
        #         break

                # Enter trade
                if ((current_high >= EntryPrice)  # current High of month is higher than ENTRY
                    # ENTRY is lower than prediction price
                    and (EntryPrice < predictions)
                    # there is not gap, of current open, openning above prev high candle
                    and (current_open <= EntryPrice)
                        and (predictions - EntryPrice > self.acceptance)):  # expected profit is above our acceptance level to filter out
                    # not profitable trades

                    ccc += 1
                    # print(ccc)
                    temp_pr = 0
                    temp_loss = 0
                    temp_profit = 0
                    trade_counter += 1
                    difference_value = 0

                    # Profit trade
                    if current_high >= predictions:
                        # debuging prints
                        #print("condition 1")
                        #print("previous high: ",prev_high)
                        #print("current high: ",current_high)
                        #print("predictions: ",predictions)
                        # record profit
                        # calculate profit (condition above) - Entry price + prediction price

                        temp_pr = - EntryPrice + predictions

                        temp_app = - EntryPrice + predictions  # Calculate  profit
                        exact_profit += temp_app

                        temp_profit = self.profit_calculation(
                            temp_pr, EntryPrice, self.budget)
                        ttl_profit += temp_profit

                        profit_trades += 1

                        #print("temp_pr: ",temp_pr)

                    if current_high < predictions and EntryPrice < predictions:  # Entry is bellow prediction price
                        #print("condition 2")
                        # record income
                        # ENTRY - current close to get real profit
                        temp_pr = - EntryPrice + current_close
                        # print(temp_pr)

                        # profit
                        if temp_pr > 0:
                            temp_profit = self.profit_calculation(
                                temp_pr, EntryPrice, self.budget)
                            ttl_profit += temp_profit
                            profit_trades += 1
                            exact_profit += temp_pr
                        # loss
                        else:
                            temp_loss = self.profit_calculation(
                                temp_pr, EntryPrice, self.budget)
                            exact_loss += temp_pr
                            ttl_loss += temp_loss

                            # Record diff value that missed to reach prediction
                            # - ENTRY + current high to get exact loss
                            difference_value = (- EntryPrice + current_high)

                            loss_trades += 1

                            # debugging prints
                            #print("\nLoss Trade: ",trade_counter)
                            #print("loss: ",temp_pr)
                            #print("\nDifference: ",difference_value)
                            #print("Current High: ",current_high)
                            #print("prediction:  ",predictions)
                            #print("Previous High: ",EntryPrice)
                            # print(df_.iloc[candle-3:candle+1,:])

                    temp_df = df_.iloc[candle -
                                       (self.window_size-1):candle+1, :].copy()

                    # Adjust revised prediction according to penalization
                    if self.penalization != 0:
                        temp_df.iloc[self.window_size-1, 8] = predictions

                    temp_df['profit'] = temp_pr

                    trades_df = pd.concat([trades_df, temp_df], axis=0)

                    profit += temp_pr
                    loss += temp_loss
                    app_profit += temp_app

                    ttl_diff += difference_value
                    ttl_diff_mean = float(ttl_diff / trade_counter)

        # Add trade counter
        counter = 1
        trades_df['trade'] = ""

        for row in range(len(trades_df)):
            trades_df.iloc[row, 11] = counter
            if trades_df.iloc[row, 8] != "nn":
                counter += 1

        print("Entry Candle: ", self.entry_candle)
        print("\nTotal Trades: ", trade_counter)
        print("Profit Trades: ", profit_trades)
        print("Loss Trades: ", loss_trades)
        print("\nWin Ratio: {} %".format(
            round(profit_trades/trade_counter, 2)*100))
        print("Loss Ratio: {} %".format(
            round(((-profit_trades/trade_counter)+1)*100), 2))
        print("\nAverage profit per trade: ", round(ttl_profit/trade_counter))
        print("\nGross profit: ", ttl_profit)
        print("Gross loss: ", ttl_loss)
        print("\nNet profit: ", ttl_profit+ttl_loss)

        if self.export_excel == True:
            trades_df.to_excel(f'{self.excel_path}/total_trades.xlsx')

        return trades_df


class GetPerformanceReport(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()
        self.entry_candle = None
        self.budget = None
        self.window_size = None
        self.export_excel = None
        self.excel_path = None

    def profit_calculation(self, difference, stock_price, budget):
        qty = round(budget/stock_price, 0)

        return round(difference * qty)

    def fit(self, entry_candle: str, budget: int, window_size: int, export_excel: bool, excel_path: str):
        """
        """

        self.entry_candle = entry_candle
        self.budget = budget
        self.window_size = window_size
        self.export_excel = export_excel
        self.excel_path = excel_path
        return self

    def transform(self, df: pd.DataFrame):
        """
        """
        performance_report = df.copy()

        try:
            performance_report = performance_report.reset_index()
            performance_report = performance_report.drop('In', axis=1)
        except:
            pass

        def GetEntryPriceColl(candle):
            if candle == 'Previous High':
                EntryPriceColl = 'High'
                EntryPriceRow = 1
            if candle == 'Current Open':
                EntryPriceColl = 'Open'
                EntryPriceRow = 0
            if candle == 'Previous Close':
                EntryPriceRow = 1
            return EntryPriceColl, EntryPriceRow

        entry_coll, entry_row = GetEntryPriceColl(self.entry_candle)

        for row in range(self.window_size-1, len(performance_report), self.window_size):
            entry = performance_report.loc[row-entry_row, entry_coll]
            difference = performance_report.loc[row, 'profit']
            prediction = performance_report.loc[row, 'prediction']
            ent_date = performance_report.loc[row-1, 'Datetime']

            # Fill data
            performance_report.loc[row, 'Entry'] = entry
            performance_report.loc[row, 'Performance'] = self.profit_calculation(
                difference, entry, self.budget)

        performance_report = performance_report.fillna("nn")

        if self.export_excel == True:
            performance_report.to_excel(
                f'{self.excel_path}/Performance_report.xlsx')

        print("--------> GetPerformanceReport completed\n")
        return performance_report


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

    # Customized loss function
    def sign_penalty(y_true, y_pred):
        penalty = 100.
        loss = tf.where(tf.less(y_true*y_pred, 0),
                        penalty * tf.square(y_true-y_pred),
                        tf.square(y_true - y_pred)
                        )

        return(tf.reduce_mean(loss, axis=-1))

    # Customized functions
    def norm_df(self, pred1):
        df_temp = pd.DataFrame()
        try:
            pred1 = pred1.drop('Date', axis=1)
        except:
            pass

        pred_np = pred1.to_numpy()
        maxv = np.max(pred_np)
        minv = np.min(pred_np)
        df_temp = (pred1.iloc[:, :]-minv)/(maxv-minv)

        return df_temp, maxv, minv

    def revert_df(self, df, maxv, minv):
        df_temp = pd.DataFrame()
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

    def fit(self, model_name: str, form_window: int, ticker: str, start_date: str, end_date: str, interval: str,
            progress: bool, condition: bool, timeperiod1: int, timeperiod2: int, timeperiod3: int, debug: bool, budget: int,
            penalization: int, acceptance: int, entry_candle: str):
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

        print()
        entry_coll, entry_row = GetEntryPriceColl(self.entry_candle)

        def Predict(pred):
            df_temp1, maxv, minv = self.norm_df(pred)
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
