import numpy as np
import tensorflow as tf
from network import get_graph
from loader import Loader

WIDTH = 256
HEIGHT = 256

def train(dataset, iterations, batch_size):
    writer = tf.summary.FileWriter(os.path.join("log", "egomotion"))
    writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(iterations):
            data, labels = dataset.get_batch(batch_size)
            x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
            y = tf.placeholder(tf.float32, [None, 5])
            #loss = get_graph(x)
            #sess.run(loss, feed_dict = {x : data, y : labels})
            print(data)
            print(labels)

train(Loader(),10,32)
