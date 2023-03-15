# Import libraries
#from final_evaluation import MakeSinglePrediction
import warnings
import logging
from final_evaluation import GetModelPerformance
from predictions_docker import prediction
from final_evaluation_docker import MakeSinglePrediction
from datetime import datetime
import pandas as pd
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

logging.getLogger('tensorflow').disabled = True

# sys.path.append('../../')

# Read variable excel
variables_df = pd.read_csv('./files/variables_df.csv', index_col=[0])
# print(variables_df)
# Extract variables
variables_dict = variables_df.to_dict()['0']
batch_size = int(variables_dict['batch_size_valid'])
window_size = int(variables_dict['window_size'])
sentiment = bool(variables_dict['sentiment'])
sentiment_type = variables_dict['sentiment_type']
ticker = variables_dict['ticker']
excel_reports = variables_dict['excel_reports']
entry_candle = variables_dict['entry_candle']
model_name = variables_dict['model_name']
model_name = f'./files/{model_name}'
indicator1 = int(variables_dict['indicator1'])
indicator2 = int(variables_dict['indicator2'])
indicator3 = int(variables_dict['indicator3'])
aggr_function = variables_dict['aggr_function']
condition = bool(variables_dict['condition'])
period = variables_dict['period']
formation_window = int(variables_dict['formation_window'])
acceptance = float(variables_dict['acceptance'])
penalization = float(variables_dict['penalization'])

# Read excels
x_test = pd.read_csv(f'./files/{ticker}_test_data.csv', index_col=[0])
x_test_x = pd.read_csv(f'./files/{ticker}_x_test_x.csv', index_col=[0])
news_df = pd.read_excel(
    f'./files/{ticker}_sentiment_analysis_final.xlsx', index_col=[0])
Dates = pd.read_csv(f'./files/{ticker}_Dates.csv', index_col=[0])
Dates = Dates.iloc[:, 0]
print("Data unpacked...")
# Run it
print("run it")
MakeSinglePrediction = MakeSinglePrediction()

fit_output = MakeSinglePrediction.fit(
    model_name=model_name,
    form_window=formation_window,
    ticker=ticker,
    start_date="2019-03-18",
    end_date="2023-03-12",
    interval=period,  # 1wk
    progress=False,
    condition=condition,
    timeperiod1=indicator1,
    timeperiod2=indicator2,
    timeperiod3=indicator3,
    debug=False,
    budget=10000,
    penalization=penalization,
    acceptance=acceptance,
    entry_candle='Current Close',
    news_df=news_df,
    sentiment=sentiment,
    sentiment_type=sentiment_type,
    sentiment_aggr=aggr_function)

# fit method outputs tuple, get only trade formation out of tuple
trade_formation = fit_output[1]


def GetDay(df):
    for date in reversed(list(df['Date'])):
        # print(date)
        date2 = date.to_pydatetime()
        day = date2.strftime('%A')
        # print(day)
        if day == "Monday":
            revised_df = df.iloc[1:, :]
            break
        else:
            revised_df = df.iloc[:-1, :]
            break
    return revised_df

print("run it2")
df = GetDay(trade_formation)
print("\nPrint Data...")
print(df)
print("\nMake prediction...")
# Make prediction
MakeSinglePrediction.transform(df)
# print("Done...")
