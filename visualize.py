import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from model import Model
from loader import TestLoader
import numpy as np

MAX_BATCH_SIZE = 16

def visualize(model_name):
    model = Model()
    sess, pred, x, training = model.load_model(model_name)

    rot_mat = np.eye(3)
    pos = np.zeros((3))
    xdata = []
    ydata = []
    plt.show()

    axes = plt.gca()
    axes.set_xlim(-200, 200)
    axes.set_ylim(-200, 200)
    line, = axes.plot(xdata, ydata, 'r-')

    def R_x(phi):
        return np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])

    def R_y(phi):
        return np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])

    def R_z(phi):
        return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    data = TestLoader(5)
    dat = data.get_test(MAX_BATCH_SIZE)
    truth = data.get_truth()
    while dat is not None:
        vec = model.predict(sess, pred, x, training, dat) #dx dy dz alfa beta gama
        for v in vec:
            d_transl = v[:3]
            d_rot_mat = np.matmul(np.matmul(R_z(v[5]), R_y(v[4])), R_x(v[3]))

            pos += np.matmul(d_transl, np.linalg.inv(rot_mat))
            rot_mat = np.matmul(rot_mat, d_rot_mat)

            xdata.append(pos[0])
            ydata.append(pos[2])
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            plt.draw()

        dat = data.get_test(MAX_BATCH_SIZE)


    plt.show()
