from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
from keras import *
from keras.layers import *
import keras


def config_session():
    config = tf.ConfigProto(device_count={'GPU': 1})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)


def load():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    return x_train, y_train, x_test, y_test


def create_model():
    input_shape = (28, 28, 1)

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    return model


def load_model():
    model_path = Path(".").absolute().parent.parent / "models" / "model.h5"
    model = keras.models.load_model(model_path.absolute().as_posix())
    return model


def train(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=1)

    evaluation = model.evaluate(x_test, y_test)
    prediction = model.predict(x_test)

    print(evaluation)
    print(prediction.argmax())


if __name__ == '__main__':
    config_session()

    x_train, y_train, x_test, y_test = load()

    # model = create_model()
    # train(model)

    model = load_model()

    for test, label in zip(x_test[:10], y_test[:10]):
        pred = model.predict(test.reshape(1, 28, 28, 1))
        print(pred.argmax(), " - ", label)

    print(test)
    print(test.shape)

    # model.save("model.h5")
