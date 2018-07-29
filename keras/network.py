import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

WIDTH_ORIG = 1226
HEIGHT_ORIG = 370

WIDTH = WIDTH_ORIG//4
HEIGHT = HEIGHT_ORIG//4

def get_network():
    input_shape = (HEIGHT, WIDTH, 3)
    num_output = 6

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_output, activation='linear'))

    model.summary()

    adam = Adam()
    model.compile(loss='mse',
                  optimizer=adam)
    return mode
