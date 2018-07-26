import os
import cv2
import numpy as np

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

        self.training_datset = []
        self.testing_dataset = []

        for id in range(0, TRAINING_SEQS):
            path = os.path.join(LABELS_DIR, labels_paths[id])
            self.training_datset += self.load(path, id)

        for id in range(TRAINING_SEQS, len(labels_paths)):
            path = os.path.join(LABELS_DIR, labels_paths[id])
            self.testing_dataset += self.load(path, id)

    def load(self, path, sequence_id):
        with open(path, "r") as file:
            dataset = []
            frame_id = 0
            for line in file:
                numbers_text = line.split()
                numbers = np.zeros(len(numbers_text))
                for i in range(len(numbers_text)):
                    numbers[i] = float(numbers_text[i])

                projection_matrix = np.reshape(numbers,(MATRIX_ROWS, MATRIX_COLUMNS))
                anno = Annotation(sequence_id, frame_id, projection_matrix)
                dataset.append(anno)

                frame_id += 1

        return dataset

class Annotation:
    def __init__(self, sequence_id, frame_id, matrix1, matrix2):
        self.sequence_id = sequence_id
        self.frame_id = frame_id

        #self.projection_matrix = np.concatenate(np.matmul(matrix1[:,:3], matrix2[:,:3]), np.matmul(matrix1[:,:3], matrix2[:,3] - matrix1[:,3])
        rotation_mat = self. rotation_mat = np.concatenate(np.matmul(matrix1[:,:3], matrix2[:,:3]))
        self.translation_mat = np.matmul(matrix1[:,:3], matrix2[:,3] - matrix1[:,3])

        #Euler's angles

        phi_x = np.arctan2(rotation_mat[2][1], rotation_mat[2][2])
        phi_y = np.arctan2(-rotation_mat[2][0], np.sqrt(rotation_mat[2][1]^2 + rotation_mat[2][2]^2))
        phi_z = np.arctan2(rotation_mat[1][0], rotation_mat[0][0])


    def get_matrix(self):
        return self.projection_matrix

    def get_image(self):
        camera1_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_0",  "{:06}.png".format(self.frame_id))
        camera2_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_1",  "{:06}.png".format(self.frame_id))

        camera1_image = cv2.imread(camera1_path)
        camera2_image = cv2.imread(camera2_path)

        camera1_path_next = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_0",  "{:06}.png".format(self.frame_id + 1))
        camera2_path_next = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_1",  "{:06}.png".format(self.frame_id + 1))

        camera1_image_next = cv2.imread(camera1_path_next)
        camera2_image_next = cv2.imread(camera2_path_next)

        camera1_image = cv2.resize(camera1_image, (WIDTH, HEIGHT))
        camera2_image = cv2.resize(camera2_image, (WIDTH, HEIGHT))
        camera1_image_next = cv2.resize(camera1_image_next, (WIDTH, HEIGHT))
        camera2_image_next = cv2.resize(camera2_image_next, (WIDTH, HEIGHT))

        return camera1_image, camera2_image, camera1_image_next, camera2_image_next

    def print_anno(self):
        print("#{0} #{1} {2}".format(self.sequence_id, self.frame_id, self.projection_matrix))
3