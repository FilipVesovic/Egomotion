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
    xdatatrue = []
    ydatatrue = []
    plt.show()

    axes = plt.gca()
    axes.set_xlim(-300, 300)
    axes.set_ylim(-300, 300)
    line, = axes.plot(xdata, ydata, 'r-')
    line2, = axes.plot(xdata, ydata, 'b-')

    def R_x(phi):
        return np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])

    def R_y(phi):
        return np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])

    def R_z(phi):
        return np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    data = TestLoader(0)
    dat = data.get_test(MAX_BATCH_SIZE)
    truth = data.get_truth()
    for tru in truth:
        trans = np.reshape(tru,(3,4))[:3,3]
        xdatatrue.append(trans[0])
        ydatatrue.append(trans[2])
        line2.set_xdata(xdatatrue)
        line2.set_ydata(ydatatrue)
    last = np.eye(4)
    while dat is not None:
        vec = model.predict(sess, pred, x, training, dat) #dx dy dz alfa beta gama
        for v in vec:
            d_transl = v[:3]
            d_rot_mat = np.matmul(np.matmul(R_z(v[5]), R_y(v[4])), R_x(v[3]))
            d_transl = np.expand_dims(d_transl, axis=1)
            mat = np.hstack([d_rot_mat,d_transl])
            mat = np.vstack([mat,[0,0,0,1]])

            next = np.matmul(last, np.linalg.inv(mat))
            last = next

            xdata.append(next[0,3])
            ydata.append(next[2,3])
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.01)

        dat = data.get_test(MAX_BATCH_SIZE)
    plt.show()
