# Import libraries
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
from final_evaluation_docker import MakeSinglePrediction
from final_evaluation_docker import GetModelPerformance
from transformers_preprocess_docker import PullData
from predictions_docker import prediction
from sentiment_docker import GetNews
from datetime import timedelta
from datetime import datetime
from datetime import date
import time

if __name__ == "__main__":

    print("\nScript started...")

    #Unpack data
    budget = 10000
    variables_df = pd.read_csv('./files/variables_df.csv', index_col=[0])
    variables_dict = variables_df.to_dict()['0']
    batch_size = int(variables_dict['batch_size_valid'])
    window_size = int(variables_dict['window_size'])
    sentiment_type = str(variables_dict['sentiment_type'])
    ticker = variables_dict['ticker']
    entry_candle = str(variables_dict['entry_candle'])
    model_name_ = str(variables_dict['model_name'])
    model_name = f'./files/{model_name_}'
    indicator1 = int(variables_dict['indicator1'])
    indicator2 = int(variables_dict['indicator2'])
    indicator3 = int(variables_dict['indicator3'])
    aggr_function = str(variables_dict['aggr_function'])
    period = str(variables_dict['period'])
    formation_window = int(variables_dict['formation_window'])
    acceptance = float(variables_dict['acceptance'])
    penalization = float(variables_dict['penalization'])
    start_date = str(variables_dict['start_date'])
    parsed_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date = parsed_date.strftime("%Y-%m-%d")
    end_date = str(variables_dict['end_date'])   
    parsed_date = datetime.strptime(end_date, "%Y-%m-%d")
    end_date = parsed_date.strftime("%Y-%m-%d")
    api_key = str(variables_dict['api_key'])  
    target_window = int(variables_dict['target_window'])
    split_ratio = float(variables_dict['split_ratio'])
    validation_ratio = float(variables_dict['validation_ratio'])
    test_ratio = float(variables_dict['test_ratio'])
    bool_list = ['shuffle','sentiment','condition','excel_reports','twitter']

    for item in bool_list:
        if str.lower(variables_dict[item]) == 'false':
            variables_dict[item] = False
        else:
            variables_dict[item] = True

    shuffle = bool(variables_dict['shuffle'])
    sentiment = bool(variables_dict['sentiment'])
    condition = bool(variables_dict['condition'])
    excel_reports = bool(variables_dict['excel_reports'])
    twitter = bool(variables_dict['twitter'])

    current_date = date.today()
    current_date = current_date.strftime('%Y-%m-%d')

    # Assuming the date is stored as a string
    date_string = current_date

    # Convert the date string to a datetime object
    date_object = datetime.strptime(date_string, '%Y-%m-%d')
    moveBack = 0

    while moveBack < 6:
        
        # Subtract two days from the datetime object
        new_date_object = date_object - timedelta(days=moveBack)

        day = new_date_object.strftime('%A')
        #print(day)
        
        if (day == 'Sunday') or (day == 'Saturday'):
            revised_date = new_date_object - timedelta(days=7)    
            revised_end_date = revised_date.strftime('%Y-%m-%d') 
            
            break
        else:
            moveBack+=1

    def GetData():
        from training_docker import SplitData

        if sentiment == True:
            GetNewsAPI = GetNews()
            GetNewsAPI.fit(ticker=ticker, start_date=start_date, end_date=revised_end_date,
                        n_news=1000, token=api_key, offset=0, export_excel=False, twitter=twitter,temp_folder=False)
            news_df = GetNewsAPI.transform()

        time.sleep(1)

        GetData = PullData()

        GetData.fit(ticker=ticker,
                    start_date=start_date,
                    end_date=revised_end_date,
                    interval=period,  # 1wk
                    progress=False,
                    condition=condition,
                    form_window=formation_window,
                    target_window=target_window,
                    timeperiod1=indicator1,#6
                    timeperiod2=indicator2,#12
                    timeperiod3=indicator3,#24
                    export_excel=False,
                    excel_path=excel_reports,
                    listed_conditions=None,
                    sentiment=sentiment,
                    sentiment_type=sentiment_type, #sentiment_type
                    news_df=news_df,
                    chart_period=period,
                    sentiment_aggr=aggr_function,
                    )

        data_prep = GetData.transform()

        time.sleep(1)
        
        # Function to check if date is Monday
        df = data_prep[data_prep['Date']!= "Month"]
        def is_monday(date):
            return date.weekday() == 0
        # Loop through first 5 rows and check if the date is Monday
        for index, row in df.head(10).iterrows():
            date_obj = row['Date'].date()
            
            if not is_monday(date_obj):
                raise Exception(f"Error: Data were not properly pulled, start date is NOT Monday. ")
    
        time.sleep(1)

        from transformers_preprocess_docker import NormalizeData

        NormalizeData = NormalizeData()

        time.sleep(1)

        NormalizeData.fit(window_size=window_size, shuffle=shuffle, debug=False,
                        export_excel=False, excel_path=excel_reports, sentiment=sentiment)

        data_normalized, Dates = NormalizeData.transform(data_prep)

        SplitData = SplitData()

        SplitData.fit(split_ratio=split_ratio, window_size=window_size,
                    dates=Dates, debug=False, export_excel=False, excel_path=excel_reports, sentiment=sentiment,validation_set=validation_ratio, test_set=test_ratio)

        _, _, x_test, _, _, x_test_x, _ = SplitData.transform(data_normalized)

        return x_test, x_test_x, Dates, news_df

    x_test, x_test_x, Dates, news_df = GetData()

    performance_df = prediction(x_test,x_test_x,news_df,Dates)

    GetModelPerformance = GetModelPerformance()

    GetModelPerformance.fit(acceptance=acceptance,
                                penalization=penalization,
                                entry_candle=entry_candle,  # Current Open
                                budget=budget,
                                window_size=window_size,
                                export_excel=False,
                                excel_path=excel_reports,
                                sentiment=sentiment)

    print("\n____________________________________________________")
    print("Summary...")
    if shuffle == True:
        print(f"Test set is sampled as {test_ratio*100}% of bellow period, data is shuffled")
        print(f'\nTotal Timeframe: {start_date} - {revised_end_date}') #initial end_date {end_date}
    else:
        print(f"Test set is sampled as {test_ratio*100}% of bellow period")
        print(f'\nTotal Timeframe: {start_date} - {end_date}')

    print(f"Data splitted as train: {split_ratio*100}%, validation: {validation_ratio*100}%, test: {test_ratio*100}%")
    print("Period: ",period)
    trades_df = GetModelPerformance.transform(performance_df)

    MakeSinglePrediction = MakeSinglePrediction()

    current_date = date.today()
    current_date = current_date.strftime('%Y-%m-%d')

    # Assuming the date is stored as a string
    date_string = current_date

    # Convert the date string to a datetime object
    date_object = datetime.strptime(date_string, '%Y-%m-%d')
    moveBack = 0

    while moveBack < 6:
        
        # Subtract two days from the datetime object
        new_date_object = date_object - timedelta(days=moveBack)

        day = new_date_object.strftime('%A')
        
        if (day == 'Sunday') or (day == 'Saturday'):
            revised_date = new_date_object.strftime('%Y-%m-%d') 
            break
        else:
            moveBack+=1
    
    fit_output = MakeSinglePrediction.fit(
        model_name=model_name,
        form_window=formation_window,
        ticker=ticker,
        start_date=start_date,
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

            if day == "Monday":
                revised_df = df.iloc[1:, :]
                break
            else:
                revised_df = df.iloc[:-1, :]
                break
        return revised_df

    print("\n____________________________________________________")
    df = GetDay(trade_formation)

    from_date = df.iloc[-1,0] + timedelta(7)
    to_date = from_date + timedelta(5)
    from_date = from_date.strftime('%Y-%m-%d')
    to_date = to_date.strftime('%Y-%m-%d')

    print("\nPrint Data...")
    print(df)
    print("\n____________________________________________________")
    print("Make prediction...")
    print("\nToday's date: ", current_date)
    print(f"Predicted period: {from_date} - {to_date}")
    print("Model name:", model_name_)
    # Make prediction
    MakeSinglePrediction.transform(df)

    print("\n")

