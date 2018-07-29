import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from network import get_network
from loader import BatchGenerator

WIDTH_ORIG = 1226
HEIGHT_ORIG = 370

WIDTH = WIDTH_ORIG//2
HEIGHT = HEIGHT_ORIG//2

LOG_DIR = 'log/'

EPOCHS_NUM = 100

def train(model,train_gen, valid_gen):

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10,
                       mode='min', verbose=1)
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)
    tb_counter  = len(os.listdir(LOG_DIR)) + 1
    tensorboard = TensorBoard(log_dir=LOG_DIR+'log_' + str(tb_counter),
                               histogram_freq=0,
                               write_graph=True,
                               write_images=False)
    callbacks = [early_stop, checkpoint, tensorboard]
    model.fit_generator(generator = train_gen,
                    steps_per_epoch  = len(train_gen),
                    epochs           = EPOCHS_NUM,
                    validation_data  = valid_gen,
                    validation_steps = len(valid_gen),
                    callbacks        = callbacks,
                    verbose          = 1,
                    workers          = 3,
                    max_queue_size   = 8)
