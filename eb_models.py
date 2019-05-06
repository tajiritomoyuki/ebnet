#coding:utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.optimizers import RMSprop

#like EXONET-XS
class Model():
    model = Sequential()
    model.add(Conv1D(16, 5, activation="relu", padding="same", input_shape=(1024, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(16, 5, activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(32, 5, activation="relu", padding="same"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation="sigmoid", input_dim=32))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

class Model1():
    model = Sequential()
    def __init__(self, conv1, conv2, conv3, kernel):
        model.add(Conv1D(conv1, kernel, activation="relu", padding="same", input_shape=(1024, 1)))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv2, kernel, activation="relu", padding="same"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv3, kernel, activation="relu", padding="same"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation="sigmoid", input_dim=conv3))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

class Model2():
    model = Sequential()
    def __init__(self, conv1, conv2, conv3, conv4, kernel):
        model.add(Conv1D(conv1, kernel, activation="relu", padding="same", input_shape=(1024, 1)))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv2, kernel, activation="relu", padding="same"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv3, kernel, activation="relu", padding="same"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv4, kernel, activation="relu", padding="same"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation="sigmoid", input_dim=conv4))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

class Model3():
    model = Sequential()
    def __init__(self, conv1, conv2, conv3, kernel):
        model.add(Conv1D(conv1, kernel, activation="relu", padding="same", input_shape=(1024, 1)))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv2, kernel, activation="relu", padding="same"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Conv1D(conv3, kernel, activation="relu", padding="same"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(conv3, activation="relu", input_dim=conv3))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#like EXONET
class Model9():
    model = Sequential()
    model.add(Conv1D(1, 5, padding="same", activation="relu", input_shape=(1024, 1)))
    model.add(Conv1D(16, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, padding="same"))
    model.add(Conv1D(16, 5, padding="same", activation="relu"))
    model.add(Conv1D(32, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, padding="same"))
    model.add(Conv1D(32, 5, padding="same", activation="relu"))
    model.add(Conv1D(64, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, padding="same"))
    model.add(Conv1D(64, 5, padding="same", activation="relu"))
    model.add(Conv1D(128, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, padding="same"))
    model.add(Conv1D(128, 5, padding="same", activation="relu"))
    model.add(Conv1D(256, 5, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, padding="same"))
    model.add(Dense(256, activation="relu", input_dim=1024))
    model.add(Dense(128, activation="relu", input_dim=256))
    model.add(Dense(64, activation="relu", input_dim=128))
    model.add(Dense(1, activation="sigmoid", input_dim=64))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
