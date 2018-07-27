import os
import numpy as np
import tensorflow as tf
from network import get_graph
from loader import Loader

WIDTH = 256
HEIGHT = 256

LOG_DIR = "log"
MODEL_DIR = "model"

class Model:

    def show_params_num(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(total_parameters)

    def train(self, dataset, epochs, iterations, batch_size):
        val_iterations = 10

        x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
        y = tf.placeholder(tf.float32, [None, 6])
        training = tf.placeholder(tf.bool)
        pred = get_graph(x, training)

        loss = tf.reduce_mean(tf.square(y - pred))

        training_summary = tf.summary.scalar("training_loss", loss)
        validation_summary = tf.summary.scalar("validation_loss", loss)

        optimizer = tf.train.AdamOptimizer(5e-4)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = optimizer.minimize(loss)

        saver = tf.train.Saver()

        idx = 0
        while(os.path.exists(os.path.join(LOG_DIR, "egomotion" +str(idx)))):
            idx += 1

        writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "egomotion" + str(idx)))
        writer.add_graph(tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.show_params_num()

        step = 0
        val_step = 0

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):

                for iter in range(iterations):
                    data, labels = dataset.get_batch(dataset.training_dataset, batch_size)
                    _, loss_value, summary = sess.run([opt, loss, training_summary], feed_dict = {x : data, y : labels, training : True})
                    writer.add_summary(summary, step)
                    step+= 1

                data, labels = dataset.get_batch(dataset.testing_dataset, batch_size)

                for iter in range(val_iterations):
                    _, val_loss_value, summary = sess.run([pred, loss, validation_summary], feed_dict = {x : data, y : labels, training : False})
                    writer.add_summary(summary, val_step)
                    val_step += 1

                saver.save(sess, os.path.join(MODEL_DIR, "model_{:05}.ckpt".format(epoch)))

    def load_model(self, model_name):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config = config)

        x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 4])
        training = tf.placeholder(tf.bool)

        pred = get_graph(x, training)

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(MODEL_DIR, model_name))
        return sess,  pred, x, training

    def predict(self, sess, pred, x, training, data):
        prediction = sess.run(pred, feed_dict = {x : data, training : False})
        return prediction
