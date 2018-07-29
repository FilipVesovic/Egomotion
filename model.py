import os
import numpy as np
import tensorflow as tf
from network import get_graph
from loader import Loader

WIDTH_ORIG = 1226
HEIGHT_ORIG = 370

WIDTH = WIDTH_ORIG//2
HEIGHT = HEIGHT_ORIG//2

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
        print('Total number of parameters:',total_parameters)

    def train(self, dataset, epochs, iterations, batch_size):
        val_iterations = 10

        x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 4], name = "x")
        y = tf.placeholder(tf.float32, [None, 6])
        training = tf.placeholder(tf.bool, name = "training")

        pred = get_graph(x, training)
        #fact = tf.exp(tf.abs(y - pred))-1
        #fact = tf.square(y - pred)
        fact = tf.abs(y - pred)
        scale = tf.constant([1., 1., 1., 1., 1., 1.])
        per_train = scale * tf.reduce_mean(fact, axis = 0)
        loss = tf.reduce_mean(per_train, axis = 0)

        training_summary = tf.summary.scalar("training_loss", loss)
        validation_summary = tf.summary.scalar("validation_loss", loss)

        translation_summary = tf.summary.scalar("translation_loss", (per_train[0]+per_train[1]+per_train[2]))
        rotation_summary = tf.summary.scalar("rotation_loss", (per_train[3]+per_train[4]+per_train[5]))

        translation_val_summary = tf.summary.scalar("translation_loss_val", (per_train[0]+per_train[1]+per_train[2]))
        rotation_val_summary = tf.summary.scalar("rotation_loss_val", (per_train[3]+per_train[4]+per_train[5]))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = optimizer.minimize(loss)

        saver = tf.train.Saver(max_to_keep=10000)

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

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            for iter in range(iterations):
                data, labels = dataset.get_batch(dataset.training_dataset, batch_size)
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    _, loss_value, summary, s2, s3 = sess.run([opt, loss, training_summary, translation_summary, rotation_summary], feed_dict = {x : data, y : labels, training : True})
                writer.add_summary(summary, step)
                writer.add_summary(s2, step)
                writer.add_summary(s3, step)

                step+= 1


            for iter in range(val_iterations):
                data, labels = dataset.get_batch(dataset.validation_dataset, batch_size)
                _, val_loss_value, summary, s2, s3 = sess.run([pred, loss, validation_summary, translation_val_summary, rotation_val_summary], feed_dict = {x : data, y : labels, training : False})
                writer.add_summary(summary, val_step)
                writer.add_summary(s2, step)
                writer.add_summary(s3, step)
                val_step += 1


            saver.save(sess, os.path.join(MODEL_DIR, "model_{:05}.ckpt".format(epoch)))
        return sess, pred, x, training

    def load_model(self, model_name, meta_name):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, meta_name))
        sess = tf.Session(config = config)

        saver.restore(sess, os.path.join(MODEL_DIR, model_name))
        graph = tf.get_default_graph()

        pred = graph.get_tensor_by_name("pred/BiasAdd:0")
        x = graph.get_tensor_by_name("x:0")
        training = graph.get_tensor_by_name("training:0")
        return sess,  pred, x, training

    def predict(self, sess, pred, x, training, data):
        prediction = sess.run(pred, feed_dict = {x : data, training : False})
        return prediction
