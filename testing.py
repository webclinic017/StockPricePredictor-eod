import tensorflow as tf
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
tf.random.set_seed(7788)
np.random.seed(7788)

# datetime object containing current date and time


def get_models(num_layers: int,
               min: int,
               max: int,
               node_step_size: int,
               features: int,
               hidden_layer_activation: str = 'selu',
               num_nodes_at_output: int = 1,
               output_layer_activation: str = 'relu') -> list:

    node_options = list(range(min, max + 1, node_step_size))
    layer_possibilities = [node_options] * num_layers
    layer_node_permutations = list(itertools.product(*layer_possibilities))

    models = []
    model_nnames = []
    for permutation in layer_node_permutations:

        kernels = [1, 10]

        for kernel in kernels:
            model = tf.keras.Sequential()
            model_name = ""

            counter_lstm = 0
            counter_cnn = 0

            for nodes_at_layer in permutation:
                if counter_cnn < 1:
                    model.add(tf.keras.layers.Conv1D(filters=nodes_at_layer, kernel_size=kernel,
                                                     strides=1, padding="same",
                                                     activation=tf.nn.selu,
                                                     input_shape=[None, features]))

                    model_name += f'CNN-nodes-{nodes_at_layer}_kernel{kernel}'

                if counter_lstm == 1 and counter_cnn == 1:
                    model.add(tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(nodes_at_layer, return_sequences=True)))
                    model_name += f'lstmBI-{nodes_at_layer}_'

                if counter_lstm == 2:
                    model.add(tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(nodes_at_layer)))
                    model_name += f'lstm-{nodes_at_layer}_'

                if counter_cnn >= 1 and counter_lstm >= 3:
                    model.add(tf.keras.layers.Dense(nodes_at_layer,
                                                    activation=hidden_layer_activation))
                    model_name += f'dense{nodes_at_layer}_'

                counter_cnn += 1
                counter_lstm += 1

            # Finalize final model
            model.add(tf.keras.layers.Dense(num_nodes_at_output,
                                            activation=output_layer_activation))

            # Get Name
            model._name = model_name[:-1]
            print(model._name)
            models.append(model)
            model_nnames.append(model._name)

    return models, model_nnames


def optimize(models: list,
             X_train: np.array,
             X_valid: np.array,
             X_test: np.array,
             labels: np.array,
             epochs: int,
             verbose: int,
             window_size: int,
             callbacks: list,
             layer: int,
             ticker: str,
             excel_path: str) -> pd.DataFrame:

    counter = 0
    now = datetime.now()
    now = now.strftime("%d%m%Y %H%M%S")

    def model_forecast1(model, series, window_size, debug):
        """
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

    def sign_penalty(y_true, y_pred):
        penalty = 100.
        loss = tf.where(tf.less(y_true*y_pred, 0),
                        penalty * tf.square(y_true-y_pred),
                        tf.square(y_true - y_pred)
                        )
        return(tf.reduce_mean(loss, axis=-1))

    result = []

    def train(model: tf.keras.Sequential, layer: int) -> dict:
        result_list = []

        optimizer2 = tf.keras.optimizers.Adam(
            learning_rate=0.0007, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        model.compile(loss=sign_penalty, optimizer=optimizer2)

        model.fit(X_train, epochs=epochs, verbose=0,
                  validation_data=X_valid, callbacks=[callbacks])

        forecast = model_forecast1(
            model, X_test, window_size=window_size, debug=False)
        result_list = sign_penalty(labels, forecast).numpy()
        dicti = {'model_name': model.name,
                 'validation_loss': result_list,
                 'layers': layer}
        # print(dicti)
        return dicti

    for model in models:
        counter += 1
        #print(model.name, end=' ... ')
        res = train(model=model, layer=layer)
        result.append(res)

        if counter == 10:
            df_final = pd.DataFrame(result)
            df_final = df_final.sort_values(
                by='validation_loss', ascending=True)
            df_final.to_excel(
                f'{excel_path}/{ticker}_{layer}_performance_{now}.xlsx')
            counter = 0

    df_final = pd.DataFrame(result)
    df_final = df_final.sort_values(by='validation_loss', ascending=True)
    return df_final
