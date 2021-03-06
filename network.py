import numpy as np
import tensorflow as tf

EPS = 0.0001

def get_graph(x, is_training):
    conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[7, 7], strides=2, padding='SAME', name='conv1', kernel_initializer=tf.contrib.layers.xavier_initializer())
    relu1 = tf.nn.relu(conv1, name='relu1')
    pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=[2, 2], strides=2, padding='VALID', name='pool1')

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', name='conv2',kernel_initializer=tf.contrib.layers.xavier_initializer())
    relu2 = tf.nn.relu(conv2, name='relu2')
    pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2, padding='VALID', name='pool2')

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=1, padding='SAME', name='conv3',kernel_initializer=tf.contrib.layers.xavier_initializer())
    relu3 = tf.nn.relu(conv3, name='relu3')
    pool3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=[2, 2], strides=2, padding='VALID', name='pool3')

#   conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', name='conv4',kernel_initializer=tf.contrib.layers.xavier_initializer())
#    relu4 = tf.nn.relu(conv4, name='relu4')
#    pool4 = tf.layers.max_pooling2d(inputs=relu4, pool_size=[2, 2], strides=2, padding='VALID', name='pool4')

#    conv5 = tf.layers.conv2d(inputs=pool4, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', name='conv5',kernel_initializer=tf.contrib.layers.xavier_initializer())
#    relu5 = tf.nn.relu(conv5, name='relu5')
#   pool5 = tf.layers.max_pooling2d(inputs=relu5, pool_size=[2, 2], strides=2, padding='VALID', name='pool5')



    #fc1_input_count = int(relu5.shape[1] * relu5.shape[2] * relu5.shape[3])
    fc1_input_count = int(pool3.shape[1] * pool3.shape[2] * pool3.shape[3])

    flat = tf.reshape(pool3, [-1, fc1_input_count], name='flat')
    fc1 = tf.layers.dense(inputs=flat, activation = tf.nn.relu , units = 128, name='fc1',kernel_initializer=tf.contrib.layers.xavier_initializer())
    drop1 = tf.layers.dropout(inputs=fc1, rate=0.5, name='drop1', training = is_training)
    fc2 = tf.layers.dense(inputs=drop1, activation = tf.nn.relu, units = 128, name='fc2',kernel_initializer=tf.contrib.layers.xavier_initializer())
    drop2 = tf.layers.dropout(inputs=fc2, rate=0.5, name='drop2', training = is_training)
    fc3 = tf.layers.dense(inputs=drop2, units = 6, kernel_initializer=tf.contrib.layers.xavier_initializer(), name = "pred")

    return fc3
