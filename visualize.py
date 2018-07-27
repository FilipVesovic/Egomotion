import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from model import load_model,predict
from loader import get_test
import numpy as np
sess, pred, x, training = load_model('model_00010.ckpt')
ysample = random.sample(range(-50, 50), 100)

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

data = get_test(10)
for dat in data:
    vec = predict(sess, np.expand_dims(dat,axis=0)) #dx dy dz alfa beta gama
    vec = vec[0]
    d_transl = vec[:3]
    d_rot_mat = np.matmul(np.matmul(R_z(vec[5]), R_y(vec[4])), R_x(vec[3]))

    pos += np.matmul(d_transl, np.linalg.inv(rot_mat))
    rot_mat = np.matmul(rot_mat, d_rot_mat)

    xdata.append(pos[0])
    ydata.append(pos[2])
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)

plt.show()
