import argparse
from loader import Loader
from train import train
from visualize import visualize
import numpy as np
from network import get_network
from keras.models import load_model

EPOCHS = 200
ITERATIONS = 100
BATCH_SIZE = 16

np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train and visualize model")
    parser.add_argument("mode")
    parser.add_argument("-model", default = None)
    args, leftovers = parser.parse_known_args()
    model = get_network()

    if args.mode == 'train':
        print("Training...")
        loader = Loader()
        train(loader.train_gen, loader.valid_gen)
        visualize(model, sess, pred, x, training, 0)

    if args.mode == 'test':
        model = load_model(args.model)
        visualize(model, sess, pred, x, training, 0)
