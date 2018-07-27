import numpy as np
import tensorflow as tf

def get_graph(x, is_training):
    conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=1, padding='SAME', name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='VALID', name='pool1')
    relu1 = tf.nn.relu(pool1, name='relu1')

    conv2 = tf.layers.conv2d(inputs=relu1, filters=32, kernel_size=[3, 3], strides=1, padding='SAME', name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='VALID', name='pool2')
    relu2 = tf.nn.relu(pool2, name='relu2')

    conv3 = tf.layers.conv2d(inputs=relu2, filters=32, kernel_size=[3, 3], strides=1, padding='SAME', name='conv3')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='VALID', name='pool3')
    relu3 = tf.nn.relu(pool3, name='relu3')

    conv4 = tf.layers.conv2d(inputs=relu3, filters=32, kernel_size=[3, 3], strides=1, padding='SAME', name='conv4')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='VALID', name='pool4')
    relu4 = tf.nn.relu(pool4, name='relu4')

    fc1_input_count = int(relu4.shape[1] * relu4.shape[2] * relu4.shape[3])
    relu4_flat = tf.reshape(relu4, [-1, fc1_input_count], name='relu4_flat')
    fc1 = tf.layers.dense(inputs=relu4_flat, activation = tf.nn.relu , units= 128, name='fc1')
    fc2 = tf.layers.dense(inputs=fc1, activation = tf.nn.relu, units= 128, name='fc2')
    fc3 = tf.layers.dense(inputs=fc2, activation = tf.nn.relu, units= 5, name='fc3')

    return fc3
