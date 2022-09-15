import numpy as np
import pandas as pd
import tensorflow as tf


def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true*y_pred, 0),
                    penalty * tf.square(y_true-y_pred),
                    tf.square(y_true - y_pred)
                    )
    return(tf.reduce_mean(loss, axis=-1))


def model1(a, b, c, d):
    model1 = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=4, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               input_shape=[None, 8]),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(a, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(b)),
        tf.keras.layers.Dense(c, activation=tf.nn.selu),
        tf.keras.layers.Dense(d, activation=tf.nn.selu),
        tf.keras.layers.Dense(1, activation=tf.nn.relu),
    ])
    return model1


def model_forecast__(model, series, window_size):
    c = 0
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size-1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(1).prefetch(1)
    forecast = model.predict(ds)
    forecast2 = np.squeeze(forecast)
    return forecast2


optimizer2 = tf.keras.optimizers.Adam(
    learning_rate=0.009, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


def training5(params, labels, x_train_tensors, x_valid_tensors, x_valid, window_size, callbacks):
    result_list = []
    params_A = []
    params_B = []
    params_C = []
    params_D = []
    params_E = []
    results_dicti = {}

    def model1(a, b, c, d, e):
        model1 = tf.keras.models.Sequential([

            tf.keras.layers.Conv1D(filters=a, kernel_size=1,
                                   strides=1, padding="same",
                                   activation=tf.nn.selu,
                                   input_shape=[None, 8]),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(b, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(c)),
            tf.keras.layers.Dense(d, activation=tf.nn.selu),
            tf.keras.layers.Dense(e, activation=tf.nn.selu),
            tf.keras.layers.Dense(1, activation=tf.nn.relu),
        ])
        return model1

    for a in params:
        for b in params:
            for c in params:
                for d in params:
                    for e in params:
                        # print stuff
                        print(f' {a} {b} {c} {d} {e}')
                        model = model1(a, b, c, d, e)
                        # train model
                        model.compile(loss=sign_penalty, optimizer=optimizer2)
                        model.fit(x_train_tensors, epochs=1200, callbacks=[
                                  callbacks], validation_data=x_valid_tensors)
                        forecast = model_forecast__(
                            model, x_valid, window_size=window_size)
                        result = sign_penalty(labels, forecast).numpy()
                        # append results
                        result_list.append(result)
                        params_A.append(a)
                        params_B.append(b)
                        params_C.append(c)
                        params_D.append(d)
                        params_E.append(e)

    results_dicti = {'model': model, 'result': result_list, 'A': params_A,
                     'B': params_B, 'C': params_C, 'D': params_D, 'E': params_E}
    temp_df = pd.DataFrame(results_dicti)
    return temp_df


def training7(params, labels, x_train_tensors, x_valid_tensors, x_valid, window_size, callbacks):
    result_list = []
    params_A = []
    params_B = []
    params_C = []
    params_D = []
    params_E = []
    params_F = []
    params_G = []

    results_dicti = {}

    def model1(a, b, c, d, e, f, g):
        model1 = tf.keras.models.Sequential([

            tf.keras.layers.Conv1D(filters=a, kernel_size=1,
                                   strides=1, padding="same",
                                   activation=tf.nn.selu,
                                   input_shape=[None, 8]),
            tf.keras.layers.Conv1D(filters=b, kernel_size=1,
                                   strides=1, padding="same",
                                   activation=tf.nn.selu,
                                   #input_shape=[None, 7]
                                   ),
            tf.keras.layers.Conv1D(filters=c, kernel_size=10,
                                   strides=1, padding="same",
                                   activation=tf.nn.selu,
                                   #input_shape=[None, 7]
                                   ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(d, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(e)),
            tf.keras.layers.Dense(f, activation=tf.nn.selu),
            tf.keras.layers.Dense(g, activation=tf.nn.selu),
            tf.keras.layers.Dense(1, activation=tf.nn.relu),
        ])
        return model1

    for a in params:
        for b in params:
            for c in params:
                for d in params:
                    for e in params:
                        for f in params:
                            for g in params:
                                # print stuff
                                print(f' {a} {b} {c} {d} {e} {f} {g}')
                                model = model1(a, b, c, d, e, f, g)
                                # train model
                                model.compile(loss=sign_penalty,
                                              optimizer=optimizer2)
                                model.fit(x_train_tensors, epochs=1200, callbacks=[
                                    callbacks], validation_data=x_valid_tensors)
                                forecast = model_forecast__(
                                    model, x_valid, window_size=window_size)
                                result = sign_penalty(labels, forecast).numpy()
                                # append results
                                result_list.append(result)
                                params_A.append(a)
                                params_B.append(b)
                                params_C.append(c)
                                params_D.append(d)
                                params_E.append(e)
                                params_F.append(f)
                                params_G.append(g)

    results_dicti = {'model': model, 'result': result_list, 'A': params_A,
                     'B': params_B, 'C': params_C, 'D': params_D, 'E': params_E, 'F': params_F, 'G': params_G}
    temp_df = pd.DataFrame(results_dicti)
    return temp_df
