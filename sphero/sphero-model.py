
import os, glob, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate

def main():
    df = load_data_files()
    if (len(df) > 5):
        print(tabulate(df.loc[:5], headers='keys', tablefmt='psql'))
    else:
        print(tabulate(df, headers='keys', tablefmt='psql'))

    features_df = df[
            [
                'speed', 'heading', 'desired_duration', 'actual_duration',
                'x0', 'y0', 'vel_x0', 'vel_y0', 'col_x', 'col_y'
            ]
        ]

    labels_df = df[['x1', 'y1', 'vel_x1', 'vel_y1']]

    # TODO: scale/normalize data and other data sciency things.
    model = build_model(len(features_df.columns), len(labels_df.columns))
    model.summary()
    _ = train_model(model, features_df.values, labels_df.values)
    save_model(model)

def load_data_files():
    script_path = os.path.realpath(os.path.dirname(__file__))
    path = os.path.join(script_path, 'out')
    all_files = glob.glob(path + "/*.csv")
    data_frame_list = []

    for data_file in all_files:
        df = pd.read_csv(data_file, index_col=None, header=0)
        data_frame_list.append(df)

    return pd.concat(data_frame_list, axis=0, ignore_index=True)

def build_model(in_shape, out_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(in_shape,)),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(out_shape)
        ]
    )

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )

    return model

def train_model(model, train_data, train_labels):
    EPOCHS = 500

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    # Callback for monitoring via TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="bin/logs/{}".format(time.time()))

    return model.fit(
                train_data, train_labels, epochs=EPOCHS,
                validation_split=0.2, verbose=0,
                callbacks=[early_stop, tensorboard])

def save_model(model):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    os.makedirs('bin/model', exist_ok=True)
    with open("bin/model/sphero-model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("bin/model/sphero-model.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    main()