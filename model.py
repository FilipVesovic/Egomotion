import numpy as np
import tensorflow as tf
from network import get_graph
from loader import Loader

WIDTH = 256
HEIGHT = 256

LOG_DIR = "log"
MODEL_DIR = "model"

def train(dataset, iterations, batch_size):
    x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
    y = tf.placeholder(tf.float32, [None, 5])
    #loss = get_graph(x)

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "egomotion"))
    writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(iterations):
            data, labels = dataset.get_batch(batch_size)

            #sess.run(loss, feed_dict = {x : data, y : labels})
            print(data)
            print(labels)
        saver.save(sess, os.path.join(MODEL_DIR, "model.ckpt"))

def predict(data):
    saver = tf.train.Saver()

    x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
    #loss = get_graph(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(MODEL_DIR, "model.ckpt"))
        #sess.run(loss, feed_dict = {x : data})


train(Loader(),10,32)
