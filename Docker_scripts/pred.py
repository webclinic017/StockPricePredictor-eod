# Import libraries
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
from final_evaluation_docker import MakeSinglePrediction
from final_evaluation_docker import GetModelPerformance
from predictions_docker import prediction
from datetime import date
from datetime import datetime
import pandas as pd
import datetime

print("\nScript started...")

# Read variable excel
variables_df = pd.read_csv('./files/variables_df.csv', index_col=[0])
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
budget = 10000
# Read excels
x_test = pd.read_csv(f'./files/{ticker}_test_data.csv', index_col=[0])
x_test_x = pd.read_csv(f'./files/{ticker}_x_test_x.csv', index_col=[0])
news_df = pd.read_excel(
    f'./files/{ticker}_sentiment_analysis_final.xlsx', index_col=[0])
Dates = pd.read_csv(f'./files/{ticker}_Dates.csv', index_col=[0])
Dates = Dates.iloc[:, 0]


# Run it
performance_df = prediction()

#print(f"ticker: {str.upper(ticker)}\n")
#print("entry candle: ",entry_candle)

GetModelPerformance = GetModelPerformance()

GetModelPerformance.fit(acceptance=acceptance,
                            penalization=penalization,
                            entry_candle=entry_candle,  # Current Open
                            budget=budget,
                            window_size=window_size,
                            export_excel=False,
                            excel_path=excel_reports,
                            sentiment=sentiment)

trades_df = GetModelPerformance.transform(performance_df)
MakeSinglePrediction = MakeSinglePrediction()

current_date = date.today()
current_date = current_date.strftime('%Y-%m-%d')

# Assuming the date is stored as a string
date_string = current_date

# Convert the date string to a datetime object
date_object = datetime.datetime.strptime(date_string, '%Y-%m-%d')
moveBack = 0

while moveBack < 6:
    
    # Subtract two days from the datetime object
    new_date_object = date_object - datetime.timedelta(days=moveBack)

    day = new_date_object.strftime('%A')
    #print(day)
    
    if (day == 'Sunday') or (day == 'Satturday'):
        revised_date = new_date_object.strftime('%Y-%m-%d') 
        break
    else:
        moveBack+=1


fit_output = MakeSinglePrediction.fit(
    model_name=model_name,
    form_window=formation_window,
    ticker=ticker,
    start_date="2019-03-18",
    end_date=revised_date,
    interval=period,  # 1wk
    progress=False,
    condition=condition,
    timeperiod1=indicator1,
    timeperiod2=indicator2,
    timeperiod3=indicator3,
    debug=False,
    budget=budget,
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

print("\n____________________________________________________")
df = GetDay(trade_formation)

from_date = df.iloc[-1,0] + datetime.timedelta(7)
to_date = from_date + datetime.timedelta(5)
from_date = from_date.strftime('%Y-%m-%d')
to_date = to_date.strftime('%Y-%m-%d')

print("\nPrint Data...")
pd.set_option('display.max_columns', None)
print(df)
print("\n____________________________________________________")
print("Make prediction...")
print("Today's date: ", current_date)
print(f"Predicted period: {from_date} - {to_date}")
# Make prediction
MakeSinglePrediction.transform(df)
#print("Penalization: ",penalization)
#print("Acceptance: ",acceptance)
print("\n")

