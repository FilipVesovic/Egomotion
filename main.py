import argparse
from loader import Loader
from model import Model
from visualize import visualize
EPOCHS = 100
ITERATIONS = 100
BATCH_SIZE = 8

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train and visualize model")
    parser.add_argument("mode", nargs = '?')
    parser.add_argument("id")
    args, leftovers = parser.parse_known_args()

    if args.mode == 'train':
        print("Training...")
        loader = Loader()
        model = Model()
        model.train(loader, EPOCHS, ITERATIONS, BATCH_SIZE)

    if args.mode == 'test':
        print("Visalize...")
        visualize('model_{:05}.ckpt'.format(args.id))
