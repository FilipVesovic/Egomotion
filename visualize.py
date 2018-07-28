import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from model import Model
from loader import TestLoader
import numpy as np
import math

MAX_BATCH_SIZE = 8

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def visualize(model_name):
    model = Model()
    sess, pred, x, training = model.load_model(model_name)

    xdata = []
    ydata = []
    xdatatrue = []
    ydatatrue = []
    plt.show()

    axes = plt.gca()
    axes.set_xlim(-300, 300)
    axes.set_ylim(-50, 550)
    line, = axes.plot(xdata, ydata, 'r-')
    line2, = axes.plot(xdata, ydata, 'b-')

    data = TestLoader(0)
    truth = data.get_truth()
    for tru in truth:
        trans = np.reshape(tru,(3,4))[:3,3]
        xdatatrue.append(trans[0])
        ydatatrue.append(trans[2])
        #line2.set_xdata(xdatatrue)
        #line2.set_ydata(ydatatrue)
    dat = data.get_test(MAX_BATCH_SIZE)
    last = np.eye(4)

    plot_numbers=[[],[],[],[],[],[]]

    count = 0

    while dat is not None:
        if(count > 280):
            break
        vec = model.predict(sess, pred, x, training, dat) #dx dy dz alfa beta gama
        for v in vec:
            for i in range(6):
                plot_numbers[i].append(v[i])

            count += 1
            d_transl = v[:3]
            d_rot_mat = eulerAnglesToRotationMatrix(v[3:])

            d_transl = np.expand_dims(d_transl, axis=1)

            mat = np.hstack([d_rot_mat,d_transl])
            mat = np.vstack([mat,[0,0,0,1]])

            next = np.matmul(last, np.linalg.inv(mat))
            last = next

            xdata.append(next[0,3])
            ydata.append(next[2,3])
            #line.set_xdata(xdata)
            #line.set_ydata(ydata)
            #plt.draw()
            #plt.pause(1e-17)
            #time.sleep(0.01)

        dat = data.get_test(MAX_BATCH_SIZE)
    for i in range(6):
        plt.plot(plot_numbers[i])
        plt.show()

    plt.show()
