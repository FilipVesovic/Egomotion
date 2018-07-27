import numpy as np
import tensorflow as tf
from network import get_graph
from loader import Loader
import os
WIDTH = 256
HEIGHT = 256

LOG_DIR = "log"
MODEL_DIR = "model"

def train(dataset, iterations, batch_size):
    x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
    y = tf.placeholder(tf.float32, [None, 5])
    training = tf.placeholder(tf.bool)
    pred = get_graph(x, training)

    loss = tf.reduce_mean(tf.square(y - pred))
    optimizer = tf.train.AdamOptimizer(5e-4)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = optimizer.minimize(loss)

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "egomotion"))
    writer.add_graph(tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(iterations):
            data, labels = dataset.get_batch(batch_size)

            _, loss_value = sess.run([opt, loss], feed_dict = {x : data, y : labels, training : True})
            if iter % 100 == 0:
                print("Loss value: {0}".format(loss_value))
                saver.save(sess, os.path.join(MODEL_DIR, "model_{:05}.ckpt".format(iter//100)))

def predict(data):
    saver = tf.train.Saver()

    x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
    pred = get_graph(x, training)

    with tf.Session() as sess:
        saver.restore(sess, os.path.join(MODEL_DIR, "model.ckpt"))
        prediction = sess.run(pred, feed_dict = {x : data, training : False})
        print(pred)

train(Loader(),10000,32)
