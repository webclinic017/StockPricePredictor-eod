<i>DISCLAIMER:
This repository is only for informative purpose, i renounce any responsibility regarding this code and trained model causing loss of money during trading on real market.</i>

# Stock Predictions
This repository enables design deep neural network model to predict stock price movement for respective stock in selected time frame. It consists of one main jupyter notebook and additional python scripts that includes objects responsible for entire pre-processing, testing and evaluating steps. It is coded in object oriented programming and it is highly customizable from main notebook.<br><br>
Jupyter notebook (01_main.ipynb) calls objects from added pyscripts, entire notebook is triggered via cells one by one in following order to 
-	Setup parameters such as ticker, period, window size etc.
-	Get sentiment analysis
-	Pull stock dataset
-	Normalize data
-	Split data in train test dataset
-	Transform data into tensors
-	Design and train neural network
-	Make predictions on test set
-	Reverse normalization to get initial prices
-	Create performance reports
-	Plot each trade from test set
-	Evaluate predictions
-	Make single prediction with most recent data

<b>Sentiment Analysis</b>
-	We use API to access https://eodhistoricaldata.com to get news and twitter tweets for respective stock. API is received in json format and already includes sentiment analysis from provider. We however add additional Vader analysis from nltk library that is applied on news titles. As result we get dataset with several sentiment analyses that can be used for model training and we can select which SA we prefer or which SA is better understandable for model.

<b>Pull stock dataset</b>
-	We use yahoo finance API to access and pull stock prices
-	We use talib library to add moving averages to dataset
-	Dataset is windowed for model training<br>
o	There is set formation window and label.<br>
o	Each row in dataset represents a candle<br>
o	Formation window represents candle formation that model sees, and label contains highest value of following candle (or candles) that model tries to predict<br>
o	For instance, chart period is 1wk (one row in dataset represents 1 trading week)<br>
o	Formation window – 4, target window - 1 (label), it means that entire dataset is split in 5 candle window, consisting of 4 candles of formation, 1 candle of label. Those windows are transformed into tensors, formed to batches and feed to neural network to train.<br>

<b>Data normalization</b>
-	We have to normalize dataset to similar scale between 0 and 1. Each window has its unique normalization and has scale between 0 – 1, therefore there is removed issue with skewed dataset (for instance, stock had in a year 2000 price 2 dollars, in 2022 it has 220 dollars, with applied normalization each window has the same scale and therefore model training will not be negatively affected.

<b>Data split</b>
-	Data are split into train and test set, train set is used purely for training of neural network, test set is used for prediction of trained network. Those predictions are later evaluated and model performance is measured.

<b>Neural network design and Model training</b>
-	Python script ‘testing.py’ includes code that can be used to ‘speed up’ the process of finding appropriate model architecture and parameter settings of neural network (credit goes to https://github.com/better-data-science). There are two customized function that via several parameters and loops will define, train and test multiple neural network architectures. This however is advised to do only with enough computational power (e.g. cloud computing).

-	Model design and training is the most challenging and time consuming part, each stock has different chart, thus neural network must be customized for each trained stock. After model is trained, it is saved as external file in repository and can be later loaded.

<b>Model prediction</b>
-	After model is trained, we use test set to predict prices on windowed formations.

<b>Reverse normalization</b>
-	Entire dataset including predictions is reverted back to initial prices.

<b>Plot data</b>
-	Each formation can be plotted and evaluated separately.

<b>Final evaluation</b>
-	Performance report is generated, entry price is set, entire profit or loss is calculated.

<b>Single prediction</b>
-	We are using trained model to make prediction for current market dataset, to predict future stock price movement.
