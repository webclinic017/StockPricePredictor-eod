import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
from transformers import pipeline


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
        self.twitter = None

    def fit(self, token: str, ticker: str, start_date: str, end_date: str, n_news: int, offset: int, export_excel: bool, twitter: bool):
        """
        """

        self.token = token
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.export_excel = export_excel
        self.n_news = n_news
        self.offset = offset
        self.twitter = twitter
        return self

    def get_customized_news1(self, stock, start_date, end_date, n_news, api_key, offset, twitter):

        if twitter == False:
            url_news = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
            news_json = requests.get(url_news).json()
            title = []
            date = []
            content = []
            sentiment = []
            for item in news_json:
                # extract data
                # print(item)
                if ((item['date'] != None) and (item['title'] != None) and (item['content'] != None) and (item['sentiment'] != None)):

                    date.append(item['date'])
                    title.append(item['title'])
                    content.append(item['content'])
                    sentiment.append(item['sentiment']['polarity'])

            dicti = {'Date': date, 'Title': title,
                     'Content': content, 'APISentiment': sentiment}

            news_df_ = pd.DataFrame(dicti)

            news_df_['Date'] = news_df_['Date'].astype('datetime64[ns]')
            news_df_['Date'] = [x.strftime("%Y-%m-%d")
                                for x in news_df_['Date']]

        if twitter == True:
            url_twitter = f'https://eodhistoricaldata.com/api/tweets-sentiments?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
            twitter_json = requests.get(url_twitter).json()
            sentiment = []
            date = []
            key = str.upper(str(stock)) + ".US"
            values = twitter_json[key]
            for i in range(len(values)):
                temp_dicti = values[i]
                date.append(temp_dicti['date'])
                sentiment.append(temp_dicti['normalized'])

            dicti = {'Date': date, 'TwitterSentiment': sentiment}

            news_df_ = pd.DataFrame(dicti)
            news_df_['Date'] = news_df_['Date'].astype('datetime64[ns]')
            news_df_['Date'] = [x.strftime("%Y-%m-%d")
                                for x in news_df_['Date']]

        return news_df_

    def transform(self):
        """
        """

        news_df = self.get_customized_news1(
            self.ticker, self.start_date, self.end_date, self.n_news, self.token, self.offset, self.twitter)

        if self.export_excel == True:
            news_df.to_excel(
                f'.\Excel reports\{self.ticker} sentiment_analysis_initial.xlsx')

        if self.twitter == False:
            # get stopwords
            nltk.download('stopwords')

            # Vader Sentiment
            nltk.downloader.download('vader_lexicon')
            news_df['VaderSentiment'] = 0
            vader = SentimentIntensityAnalyzer()

            # sentiment-roberta-large-english
            # news_df['RobertaLargeSentiment'] = 0
            # news_df['RobertaLargeLabel'] = 0

            # RobertaLarge = pipeline(
            #     "sentiment-analysis", model="siebert/sentiment-roberta-large-english")

            for row in range(news_df.shape[0]):
                title = news_df.iloc[row, 1]
                title = f'"{title}"'
                content = news_df.iloc[row, 2]
                score = vader.polarity_scores(title)
                compound_score = score['compound']
                news_df.loc[row, 'VaderSentiment'] = compound_score

                # output_roberta = RobertaLarge(title)
                # score_roberta = output_roberta[0]['score']
                # label_roberta = output_roberta[0]['label']
                # news_df.loc[row, 'RobertaLargeSentiment'] = score_roberta
                # news_df.loc[row, 'RobertaLargeLabel'] = label_roberta

            # calculate combined sentiment
            for row in range(news_df.shape[0]):
                news_df.loc[row, 'CombinedVaderSentiment'] = np.mean(
                    news_df.loc[row, 'VaderSentiment'] + news_df.loc[row, 'APISentiment'])

        # sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

        if self.export_excel == True:
            news_df.to_excel(
                f'.\Excel reports\{self.ticker} sentiment_analysis_final.xlsx')
            news_df.to_excel(
                f'.\Temp\{self.ticker}_sentiment_analysis_final.xlsx')

        print("--------> GetNews completed\n")
        return news_df
