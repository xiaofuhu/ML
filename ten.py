import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from tqdm import tqdm

def distribute_dataset(train_to_test, x_train, y_train, x_test, y_test):
    # train to test. test to train if negative
    def swap(num, a, b):
        # num elements from a to b
        extra = a[0 - num:len(a)]
        a = a[0:0 - num]
        b = np.concatenate((b, extra))
        return a, b

    if train_to_test >= 0:
        x_train, x_test = swap(train_to_test, x_train, x_test)
        y_train, y_test = swap(train_to_test, y_train, y_test)
        return x_train, y_train, x_test, y_test
    else:
        x_test, x_train = swap(0 - train_to_test, x_test, x_train)
        y_test, y_train = swap(0 - train_to_test, y_test, y_train)
        return x_train, y_train, x_test, y_test

def visualize_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.show()

def arr_rgb2gray(arr):
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    new_arr = np.empty((0,32,32))
    for i in tqdm(range(len(arr))):
        elem = rgb2gray(arr[i])
        new_arr = np.append(new_arr, [elem], axis=0)
    new_arr = np.expand_dims(new_arr, axis=3)
    return new_arr

def shorten_arr(percentage, arr):
    num = int(len(arr) * percentage / 100)
    return arr[0:num]

if __name__ == "__main__":
    print("======================\n")
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        x_train = np.load("./data/x_train.npy")
        x_test = np.load("./data/x_test.npy")
        y_train = np.load("./data/y_train.npy")
        y_test = np.load("./data/y_test.npy")
    else:
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train),(x_test, y_test) = cifar10.load_data()
        PERC = 25
        x_train, y_train, x_test, y_test = \
            shorten_arr(PERC, x_train), shorten_arr(PERC, y_train), \
                shorten_arr(PERC, x_test), shorten_arr(PERC, y_test)
        x_train = arr_rgb2gray(x_train)
        print(x_train[0].shape)
        x_test = arr_rgb2gray(x_test)
        # save
        np.save("./data/x_train.npy", x_train)
        np.save("./data/x_test.npy", x_test)
        np.save("./data/y_train.npy", y_train)
        np.save("./data/y_test.npy", y_test)

    # x_train, y_train, x_test, y_test = distribute_dataset(-500, x_train, y_train, x_test, y_test)
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(32, 32)),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # ])
    print("======================\n")
    model.summary()
    print("\n======================")
    # model.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_split=0.10, epochs=100)
    model.evaluate(x_test, y_test)
    visualize_history(history)