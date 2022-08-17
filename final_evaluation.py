from datetime import datetime
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
import numpy as np


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

        print("Done")
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
        self.budget = None
        self.export_excel = None

    def profit_calculation(self, difference, stock_price, budget):
        qty = round(budget/stock_price, 0)

        return round(difference * qty)

    def fit(self, acceptance: int, penalization: int, entry_candle: str, budget: int, export_excel: bool):
        """
        """
        self.acceptance = acceptance
        self.penalization = penalization
        self.entry_candle = entry_candle
        self.budget = budget
        self.export_excel = export_excel

        return self

    def transform(self, df: pd.DataFrame):
        """
        """

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

                    temp_df = df_.iloc[candle-24:candle+1, :].copy()

                    # Adjust revised prediction according to penalization
                    if self.penalization != 0:
                        temp_df.iloc[5, 8] = predictions

                    temp_df['profit'] = temp_pr

                    trades_df = pd.concat([trades_df, temp_df], axis=0)

                    profit += temp_pr
                    loss += temp_loss
                    app_profit += temp_app

                    ttl_diff += difference_value
                    ttl_diff_mean = float(ttl_diff / trade_counter)

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
            trades_df.to_excel('total_trades.xlsx')

        return trades_df
