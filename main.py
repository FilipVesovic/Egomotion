import argparse
from loader import Loader
from model import Model
from visualize import visualize
import numpy as np

EPOCHS = 20
ITERATIONS = 100
BATCH_SIZE = 8

np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train and visualize model")
    parser.add_argument("mode")
    parser.add_argument("-id", default = 0)
    args, leftovers = parser.parse_known_args()
    model = Model()

    if args.mode == 'train':
        print("Training...")
        loader = Loader()
        sess, pred, x, training = model.train(loader, EPOCHS, ITERATIONS, BATCH_SIZE)
        visualize(model, sess, pred, x, training)

    if args.mode == 'test':
        idx = int(args.id)
        print("Visalize {0}...".format(idx))
        sess, pred, x, training = model.load_model('model_{:05}.ckpt'.format(idx),'model_{:05}.ckpt.meta'.format(idx))
        visualize(model, sess, pred, x, training)
