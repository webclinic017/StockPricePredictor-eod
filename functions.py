import numpy as np
import tensorflow as tf
from datetime import timedelta
from datetime import datetime
from datetime import date
import time

def model_forecast(model, series, window_size, debug):
    """
    Get model, data and window size as an input. 
    Make prediction window is subtracted by 1, since we do not need label in window, 
    label value is skipped
    """
    c = 0

    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size-1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))

    if debug == True:
        # This block of code will print out data on which is made prediction
        for item in ds:
            c += 1
            if c < 3:
                print("\n"+str(c) + " prediction:\n ", item)
            else:
                break

    ds = ds.batch(1).prefetch(1)
    forecast = model.predict(ds)
    forecast2 = np.squeeze(forecast)
    return forecast2

def GetCurrentDate():
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
            revised_date = new_date_object - timedelta(days=0)    
            revised_end_date = revised_date.strftime('%Y-%m-%d') 
            
            break
        else:
            moveBack+=1
    print("End Date: ", revised_end_date)
    return revised_end_date