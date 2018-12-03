"""Training a NN to approximate half circle function centered at origin."""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time

def generate_data(num_data_points):
    # [x, y, center_x, center_y]
    data = np.random.random_sample((num_data_points, 4))*5
    # radius = sqrt((x - center_x)^2 + (y - center_y)^2)
    labels = np.sqrt(np.square(data[:, 0] - data[:, 2]) + np.square(data[:, 1] - data[:, 3]))
    return data, labels

def build_model(in_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(in_shape,)),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae'])

    return model

def train_model(model, train_data, train_labels):
    """Train the model"""
    EPOCHS = 500

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    # Callback for monitoring via TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="bin/logs/{}".format(time()))

    return model.fit(
                train_data, train_labels, epochs=EPOCHS,
                validation_split=0.2, verbose=0,
                callbacks=[early_stop, PrintDot(), tensorboard])

def evaluate_model(model, test_data, test_labels):
    """Evaluates the model against set of test data"""
    [loss, mean_abs_error] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:7.2f}".format(mean_abs_error))

    test_predictions = model.predict(test_data).flatten()
    plot_predictions_and_truth(test_predictions, test_labels)
    
    error = test_predictions - test_labels
    plot_error(error)

def save_model(model):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    os.makedirs('bin/model', exist_ok=True)
    with open("bin/model/circle-radius-model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("bin/model/circle-radius-model.h5")
    print("Saved model to disk")

# Simple calback to illustrate how callbacks work.
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])

def plot_predictions_and_truth(test_predictions, test_labels):
    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])

def plot_error(error):
    plt.figure()
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()

def main():
    NUM_DATA_POINTS = 1000
    NUM_TEST_DATA_POINTS = 200
    train_data, train_labels = generate_data(NUM_DATA_POINTS)
    test_data, test_labels = generate_data(NUM_TEST_DATA_POINTS)

    model = build_model(train_data.shape[1])
    model.summary()

    history = train_model(model, train_data, train_labels)
    plot_history(history)

    evaluate_model(model, test_data, test_labels)

    save_model(model)

if __name__ == '__main__':
    main()
