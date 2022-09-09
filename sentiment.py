import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests


class GetNews(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.token = None
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.n_news = None
        self.offset = None
        self.export_excel = None

    def fit(self, token: str, ticker: str, start_date: str, end_date: str, n_news: int, offset: int, export_excel: bool):
        """
        """

        self.token = token
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.export_excel = export_excel
        self.n_news = n_news
        self.offset = offset

        return self

    def get_customized_news1(self, stock, start_date, end_date, n_news, api_key, offset):
        url_news = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
        #url_twitter = f'https://eodhistoricaldata.com/api/tweets-sentiments?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
        news_json = requests.get(url_news).json()
        title = []
        date = []
        content = []
        sentiment = []

        for item in news_json:
            # extract data
            date.append(item['date'])
            title.append(item['title'])
            content.append(item['content'])
            sentiment.append(item['sentiment']['polarity'])

        dicti = {'Date': date, 'Title': title,
                 'Content': content, 'APISentiment': sentiment}

        news_df_ = pd.DataFrame(dicti)

        news_df_['Date'] = news_df_['Date'].astype('datetime64[ns]')
        news_df_['Date'] = [x.strftime("%Y-%m-%d") for x in news_df_['Date']]

        return news_df_

    def transform(self):
        """
        """

        news_df = self.get_customized_news1(
            self.ticker, self.start_date, self.end_date, self.n_news, self.token, self.offset)

        nltk.downloader.download('vader_lexicon')

        news_df['VaderSentiment'] = 0
        vader = SentimentIntensityAnalyzer()

        for row in range(news_df.shape[0]):
            title = news_df.iloc[row, 1]
            title = f'"{title}"'
            content = news_df.iloc[row, 2]
            score = vader.polarity_scores(title)
            compound_score = score['compound']
            news_df.iloc[row, 4] = compound_score

        # calculate combined sentiment
        for row in range(news_df.shape[0]):
            news_df.loc[row, 'CombinedSentiment'] = np.mean(
                news_df.loc[row, 'VaderSentiment'] + news_df.loc[row, 'APISentiment'])

        if self.export_excel == True:
            news_df.to_excel('sentiment_analysis.xlsx')

        print("--------> GetNews completed\n")
        return news_df
