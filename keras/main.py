import argparse
from loader import Loader
from train import train
from visualize import visualize
import numpy as np
from network import get_network
from keras.models import load_model
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())

np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train and visualize model")
    parser.add_argument("mode")
    parser.add_argument("-model", default = None)
    parser.add_argument("-seq", default = 0)
    args, leftovers = parser.parse_known_args()
    model = get_network()

    if args.mode == 'train':
        print("Training...")
        loader = Loader()
        train(model, loader.train_gen, loader.valid_gen)
        visualize(model, 0)

    if args.mode == 'test':
        print("Loading", str(args.model))
        model = load_model(str(args.model))
        visualize(model, int(args.seq))
