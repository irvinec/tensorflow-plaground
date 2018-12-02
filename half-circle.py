"""Training a NN to approximate half circle function centered at origin."""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_data_points):
    data = np.empty((num_data_points, 2))
    labels = np.empty(num_data_points)
    for index in range(num_data_points):
        # radius [1, 5)
        radius = np.random.random_sample()*4 + 1
        # x [-radius, radius)
        x = np.random.random_sample()*2*radius - radius
        # y = sqrt(r^2 - x^2)
        y = np.sqrt(np.square(radius) - np.square(x))
        data[index] = [x, radius]
        labels[index] = y

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

def main():
    train_data, train_labels = generate_data(500)
    test_data, test_labels = generate_data(200)
    model = build_model(train_data.shape[1])
    model.summary()

    # Train the model
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    EPOCHS = 500
    history = model.fit(
                train_data, train_labels, epochs=EPOCHS,
                validation_split=0.2, verbose=0,
                callbacks=[early_stop, PrintDot()])

    plot_history(history)

    # Evaluate the model
    [loss, mean_abs_error] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:7.2f}".format(mean_abs_error))

    # Make some predictions
    test_predictions = model.predict(test_data).flatten()

    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])

    error = test_predictions - test_labels

    plt.figure()
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()

if __name__ == '__main__':
    main()
