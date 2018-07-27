import os
import cv2
import numpy as np
from random import shuffle
import math
import matplotlib.pyplot as plt

MATRIX_ROWS = 3
MATRIX_COLUMNS = 4

LABELS_DIR = os.path.join("dataset", "poses")
DATASET_DIR = os.path.join("dataset", "sequences")

TRAINING_SEQS = 10

WIDTH = 256
HEIGHT = 256

class Loader:
    def __init__(self):
        labels_paths = os.listdir(LABELS_DIR)
        labels_paths = sorted(labels_paths)

        self.training_dataset = []
        self.validation_dataset = []
        self.testing_dataset = []

        for id in range(0, TRAINING_SEQS):
            path = os.path.join(LABELS_DIR, labels_paths[id])
            self.training_dataset += self.load(path, id)

        for id in range(TRAINING_SEQS, len(labels_paths)):
            path = os.path.join(LABELS_DIR, labels_paths[id])
            self.testing_dataset += self.load(path, id)

#        shuffle(self.training_dataset)

        valid_split = int(len(self.training_dataset) * 0.8)
        self.training = self.training_dataset
        self.training_dataset = self.training[:valid_split]
        self.validation_dataset = self.training[valid_split:]

        self.visualize(self.training_dataset[:2000])

        print("Training set size: ", len(self.training_dataset))
        print("Validation set size: ", len(self.validation_dataset))
        print("Testing set size: ", len(self.testing_dataset))

    def visualize(self, dataset):
        plot_numbers=[[],[],[],[],[],[]]
        for data in dataset:
            for i in range(6):
                plot_numbers[i].append(data.get_matrix()[i])
        for i in range(6):
            plt.plot(plot_numbers[i])
            plt.show()


    def load(self, path, sequence_id):
        with open(path, "r") as file:
            dataset = []
            frame_id = 0
            last_matrix = None

            for line in file:
                numbers_text = line.split()
                numbers = np.zeros(len(numbers_text))
                for i in range(len(numbers_text)):
                    numbers[i] = float(numbers_text[i])
            #    print(line)
                projection_matrix = np.reshape(numbers,(MATRIX_ROWS, MATRIX_COLUMNS))
            #    print(projection_matrix)
                if(last_matrix is not None):
                    anno = Annotation(sequence_id, frame_id - 1, last_matrix, projection_matrix)
                    dataset.append(anno)
                last_matrix = projection_matrix

                frame_id += 1

        return dataset

    def get_batch(self, data, batch_size):
        batch_ids = np.random.randint(0, len(data), size = batch_size)
        imgs = None
        labels = None
        for id in batch_ids:
            if(imgs is None):
                imgs = np.expand_dims(data[id].get_image(), axis = 0)
            else:
                imgs = np.concatenate((imgs, np.expand_dims(data[id].get_image(), axis = 0)),axis = 0)
            if(labels is None):
                labels = np.expand_dims(data[id].get_matrix(), axis = 0)
            else:
                labels = np.concatenate((labels, np.expand_dims(data[id].get_matrix(), axis = 0)),axis = 0)
        return imgs, labels

def get_test(sequence):
    frame_id = 0
    dataset = None
    while frame_id < 500 and os.path.exists(os.path.join(DATASET_DIR,  "{:02}".format(sequence), "image_0",  "{:06}.png".format(frame_id + 1))):
        camera1_path = os.path.join(DATASET_DIR,  "{:02}".format(sequence), "image_0",  "{:06}.png".format(frame_id))
        camera2_path = os.path.join(DATASET_DIR,  "{:02}".format(sequence), "image_1",  "{:06}.png".format(frame_id))

        camera1_image = cv2.imread(camera1_path, 0)
        camera2_image = cv2.imread(camera2_path, 0)

        camera1_path_next = os.path.join(DATASET_DIR,  "{:02}".format(sequence), "image_0",  "{:06}.png".format(frame_id + 1))
        camera2_path_next = os.path.join(DATASET_DIR,  "{:02}".format(sequence), "image_1",  "{:06}.png".format(frame_id + 1))

        camera1_image_next = cv2.imread(camera1_path_next, 0)
        camera2_image_next = cv2.imread(camera2_path_next, 0)

        camera1_image = cv2.resize(camera1_image, (WIDTH, HEIGHT))
        camera2_image = cv2.resize(camera2_image, (WIDTH, HEIGHT))
        camera1_image_next = cv2.resize(camera1_image_next, (WIDTH, HEIGHT))
        camera2_image_next = cv2.resize(camera2_image_next, (WIDTH, HEIGHT))

        frame = np.concatenate([np.expand_dims(camera1_image,axis=2),np.expand_dims(camera2_image,axis=2),np.expand_dims(camera1_image_next,axis=2),np.expand_dims(camera2_image_next,axis=2)],axis=2)
        if dataset is None:
            dataset = np.expand_dims(frame, axis = 0)
        else:
            dataset = np.concatenate((dataset, np.expand_dims(frame, axis = 0)))
        frame_id += 1
    return dataset

class Annotation:
    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    def __init__(self, sequence_id, frame_id, matrix1, matrix2):
        self.sequence_id = sequence_id
        self.frame_id = frame_id

        matrix1 = np.vstack([matrix1, [0,0,0,1]])
        matrix2 = np.vstack([matrix2, [0,0,0,1]])
    #    print(matrix1)
    #print(matrix2)
        #posa1*inv pose0
        rotation = np.matmul(np.linalg.inv(matrix2),matrix1)
        #rotation = np.matmul(matrix2,np.linalg.inv(matrix1))
        #rotation = np.matmul(np.linalg.inv(matrix1),matrix2)
    #    print(rotation)
        self.translation_mat = rotation[0:3,3]
    #    print(self.translation_mat)
        #v = np.matmul(np.matmul(np.array([0, 0, 1]), matrix2[:,:3]), np.linalg.inv(matrix1[:,:3]))

        v = self.rotationMatrixToEulerAngles(rotation[:3,:3])

        self.x = v[0]
        self.y = v[1]
        self.z = v[2]

    def get_matrix(self):
        return np.array([self.translation_mat[0], self.translation_mat[1], self.translation_mat[2], self.x, self.y, self.z])

    def get_image(self):
        camera1_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_0",  "{:06}.png".format(self.frame_id))
        camera2_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_1",  "{:06}.png".format(self.frame_id))

        camera1_image = cv2.imread(camera1_path, 0)
        camera2_image = cv2.imread(camera2_path, 0)

        camera1_path_next = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_0",  "{:06}.png".format(self.frame_id + 1))
        camera2_path_next = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_1",  "{:06}.png".format(self.frame_id + 1))

        camera1_image_next = cv2.imread(camera1_path_next, 0)
        camera2_image_next = cv2.imread(camera2_path_next, 0)

        camera1_image = cv2.resize(camera1_image, (WIDTH, HEIGHT))
        camera2_image = cv2.resize(camera2_image, (WIDTH, HEIGHT))
        camera1_image_next = cv2.resize(camera1_image_next, (WIDTH, HEIGHT))
        camera2_image_next = cv2.resize(camera2_image_next, (WIDTH, HEIGHT))

        return np.concatenate([np.expand_dims(camera1_image,axis=2),np.expand_dims(camera2_image,axis=2),np.expand_dims(camera1_image_next,axis=2),np.expand_dims(camera2_image_next,axis=2)],axis=2)

    def print_anno(self):
        print("#{0} #{1} {2}".format(self.sequence_id, self.frame_id, self.get_matrix()))
