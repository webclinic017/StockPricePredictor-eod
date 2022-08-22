import tensorflow as tf


def mrk_model():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=8, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               input_shape=[None, 7]),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               #input_shape=[None, 7]
                               ),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               #input_shape=[None, 7]
                               ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(9, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(9)),
        tf.keras.layers.Dense(4, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(2, activation=tf.nn.selu),
        tf.keras.layers.Dense(1, activation=tf.nn.relu),
    ])

    # # optimizer2 = tf.keras.optimizers.Adam(
    # # learning_rate=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    # optimizer5 = tf.keras.optimizers.Adagrad(
    #     learning_rate=0.005, initial_accumulator_value=5, epsilon=1e-07, name='Adagrad')

    # model.compile(loss=sign_penalty,
    #               optimizer=optimizer5,
    #               )

    # model.fit(x_train_tensors, epochs=1200, callbacks=[
    #     callbacks], validation_data=x_valid_tensors)
    return model


def clb_model():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=8, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               input_shape=[None, 7]),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               #input_shape=[None, 7]
                               ),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               #input_shape=[None, 7]
                               ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(9, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(9)),

        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(2, activation=tf.nn.selu),
        tf.keras.layers.Dense(1, activation=tf.nn.relu),
    ])
    # optimizer2 = tf.keras.optimizers.Adam(
    #     learning_rate=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    # optimizer5 = tf.keras.optimizers.Adagrad(
    #     learning_rate=0.005, initial_accumulator_value=5, epsilon=1e-07, name='Adagrad')

    # model.compile(loss=sign_penalty,
    #               optimizer=optimizer5,
    #               )

    # model.fit(x_train_tensors, epochs=1200, callbacks=[
    #     callbacks], validation_data=x_valid_tensors)
    return model


def stne_model():
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv1D(filters=8, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               input_shape=[None, 7]),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               #input_shape=[None, 7]
                               ),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10,
                               strides=1, padding="same",
                               activation=tf.nn.selu,
                               #input_shape=[None, 7]
                               ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12)),

        tf.keras.layers.Dense(4, activation=tf.nn.selu),
        tf.keras.layers.Dense(4, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(3, activation=tf.nn.selu),
        tf.keras.layers.Dense(2, activation=tf.nn.selu),
        tf.keras.layers.Dense(1, activation=tf.nn.relu),
    ])

    # optimizer2 = tf.keras.optimizers.Adam(
    #     learning_rate=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    # optimizer5 = tf.keras.optimizers.Adagrad(
    #     learning_rate=0.005, initial_accumulator_value=5, epsilon=1e-07, name='Adagrad')

    # model.compile(loss=sign_penalty,
    #             optimizer=optimizer5,
    #             )

    # model.fit(x_train_tensors, epochs=1200, callbacks=[
    #         callbacks], validation_data=x_valid_tensors)
    return model
