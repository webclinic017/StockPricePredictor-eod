
def GetData():
    
    from training_docker import SplitData
    from training_docker import GetTensoredDataset
    from training_docker import GetTensoredDataset
    from transformers_preprocess_docker import PullData
    import pandas as pd
    import time
    from sentiment_docker import GetNews
    from datetime import datetime
    variables_df = pd.read_csv('./files/variables_df.csv', index_col=[0])
    # Extract variables
    variables_dict = variables_df.to_dict()['0']
    #print(variables_dict)
    
    batch_size = int(variables_dict['batch_size_valid'])
    window_size = int(variables_dict['window_size'])
    
    sentiment_type = str(variables_dict['sentiment_type'])
    ticker = variables_dict['ticker']
    entry_candle = str(variables_dict['entry_candle'])
    model_name = str(variables_dict['model_name'])
    model_name = f'./files/{model_name}'
    indicator1 = int(variables_dict['indicator1'])
    indicator2 = int(variables_dict['indicator2'])
    indicator3 = int(variables_dict['indicator3'])
    aggr_function = str(variables_dict['aggr_function'])
    
    period = str(variables_dict['period'])
    formation_window = int(variables_dict['formation_window'])
    acceptance = float(variables_dict['acceptance'])
    penalization = float(variables_dict['penalization'])
    

    start_date = variables_dict['start_date']
    parsed_date = datetime.strptime(start_date, "%d/%m/%Y")
    start_date = parsed_date.strftime("%Y-%m-%d")
    end_date = variables_dict['end_date']
    
    parsed_date = datetime.strptime(end_date, "%d/%m/%Y")
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
   
    budget = 10000
    
    # print("start: ",start_date)
    # print("end: ",end_date)

    if sentiment == True:

        GetNewsAPI = GetNews()

        GetNewsAPI.fit(ticker=ticker, start_date=start_date, end_date=end_date,
                    n_news=1000, token=api_key, offset=0, export_excel=False, twitter=twitter,temp_folder=False)
        news_df = GetNewsAPI.transform()

    time.sleep(1)

    GetData = PullData()

    GetData.fit(ticker=ticker,
                start_date=start_date,
                end_date=end_date,
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

    from transformers_preprocess_docker import NormalizeData

    NormalizeData = NormalizeData()

    time.sleep(1)

    NormalizeData.fit(window_size=window_size, shuffle=shuffle, debug=False,
                    export_excel=False, excel_path=excel_reports, sentiment=sentiment)

    data_normalized, Dates = NormalizeData.transform(data_prep)

    SplitData = SplitData()

    SplitData.fit(split_ratio=split_ratio, window_size=window_size,
                dates=Dates, debug=False, export_excel=False, excel_path=excel_reports, sentiment=sentiment,validation_set=validation_ratio, test_set=test_ratio)

    x_train, x_valid, x_test, x_train_x, x_valid_x, x_test_x,_ = SplitData.transform(data_normalized)

    return x_test, x_test_x, Dates, news_df