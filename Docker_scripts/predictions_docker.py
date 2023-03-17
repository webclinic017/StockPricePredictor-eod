def prediction():
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    from functions_docker import model_forecast

    # Read excel with variables
    variables_df = pd.read_csv('./files/variables_df.csv', index_col=[0])

    variables_dict = variables_df.to_dict()['0']
    batch_size = int(variables_dict['batch_size_valid'])
    window_size = int(variables_dict['window_size'])
    sentiment = bool(variables_dict['sentiment'])
    sentiment_type = variables_dict['sentiment_type']
    ticker = variables_dict['ticker']
    excel_reports = variables_dict['excel_reports']
    entry_candle = variables_dict['entry_candle']
    model_name = variables_dict['model_name']
    indicator1 = int(variables_dict['indicator1'])
    indicator2 = int(variables_dict['indicator2'])
    indicator3 = int(variables_dict['indicator3'])
    aggr_function = variables_dict['aggr_function']
    condition = bool(variables_dict['condition'])
    period = variables_dict['period']
    formation_window = int(variables_dict['formation_window'])

    # Read excels
    x_test = pd.read_csv(f'./files/{ticker}_test_data.csv', index_col=[0])
    x_test_x = pd.read_csv(f'./files/{ticker}_x_test_x.csv', index_col=[0])
    news_df = pd.read_excel(
        f'./files/{ticker}_sentiment_analysis_final.xlsx', index_col=[0])
    Dates = pd.read_csv(f'./files/{ticker}_Dates.csv', index_col=[0])
    Dates = Dates.iloc[:, 0]

    # Load model

    model_name = variables_dict['model_name']

    def sign_penalty(y_true, y_pred):
        penalty = 100.
        loss = tf.where(tf.less(y_true*y_pred, 0),
                        penalty * tf.square(y_true-y_pred),
                        tf.square(y_true - y_pred)
                        )

        return(tf.reduce_mean(loss, axis=-1))

    tf.keras.losses.sign_penalty = sign_penalty
    model = tf.keras.models.load_model(f'./files/{model_name}', custom_objects={
        'sign_penalty': sign_penalty})
    

    forecast = model_forecast(
        model, x_test, window_size=window_size, debug=False)

    from training_docker import GetTensoredDataset

    GetTensoredValidDataset = GetTensoredDataset()

    GetTensoredValidDataset.fit(
        window_size=window_size, batch_size=batch_size, train=False, debug=False)

    x_test_tensors, labels = GetTensoredValidDataset.transform(x_test)

    from transformers_preprocess_docker import ReverseNormalization

    ReverseNormalization = ReverseNormalization()

    ReverseNormalization.fit(forecasts=forecast, labels=labels,
                             x_test=x_test, x_test_x=x_test_x, window_size=window_size, debug=False,
                             sentiment=sentiment, sentiment_type=sentiment_type)

    df = ReverseNormalization.transform()

    from final_evaluation_docker import GetFinalDataframe

    GetFinalDataframe = GetFinalDataframe()

    GetFinalDataframe.fit(dates=Dates,
                          x_test=x_test,
                          sentiment=sentiment,
                          sentiment_type=sentiment_type)

    reversed_df = GetFinalDataframe.transform(df)

    return reversed_df
