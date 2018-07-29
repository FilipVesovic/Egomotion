import numpy as np
import cv2
np.random.seed(0)
from keras.utils import Sequence

FRAMES = 2
MATRIX_ROWS = 3
MATRIX_COLUMNS = 4

LABELS_DIR = os.path.join("dataset", "poses")
DATASET_DIR = os.path.join("dataset", "sequences")

TRAINING_SEQS = 10

WIDTH_ORIG = 1226
HEIGHT_ORIG = 370

WIDTH = WIDTH_ORIG//2
HEIGHT = HEIGHT_ORIG//2

class BatchGenerator(Sequence):
    def __init__(self,
                 annotations,
                 batch_size):
        self.batch_size = batch_size
        self.annotations = annotations
        self.counter = 0

    def __len__(self):
        return int(len(self.annotations) / self._batch_size)

    def __getitem__(self, idx):
        x_batch = []
        y_batch= []
        for i in range(self.batch_size):
            x_batch.append(self.annotations.get_image())
            y_batch.append(self.annotations.get_matrix())

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        self.counter += 1
        return x_batch, y_batch

    def on_epoch_end(self):
        self.annotations.shuffle()
        self.counter = 0

class Loader:
    def __init__(self):
        labels_paths = os.listdir(LABELS_DIR)
        labels_paths = sorted(labels_paths)

        self.training_dataset = []
        self.validation_dataset = []

        for id in range(0, TRAINING_SEQS):
            path = os.path.join(LABELS_DIR, labels_paths[id])
            self.training_dataset += self.load(path, id)

        valid_split = int(len(self.training_dataset) * 0.8)
        self.training = self.training_dataset
        self.training_dataset = self.training[:valid_split]
        self.validation_dataset = self.training[valid_split:]

        print("Training set size: ", len(self.training_dataset))
        print("Validation set size: ", len(self.validation_dataset))

        self.train_gen =  BatchGenerator(self.training_dataset, BATCH_SIZE)
        self.valid_gen = BatchGenerator(self.validation_dataset, BATCH_SIZE)

    def load(self, path, sequence_id):
        with open(path, "r") as file:
            dataset = []
            frame_id = 0
            last = []

            for line in file:
                numbers_text = line.split()
                numbers = np.zeros(len(numbers_text))
                for i in range(len(numbers_text)):
                    numbers[i] = float(numbers_text[i])

                projection_matrix = np.reshape(numbers,(MATRIX_ROWS, MATRIX_COLUMNS))
                last.append(projection_matrix)

                if(len(last) >= FRAMES):
                    anno = Annotation(sequence_id, frame_id - FRAMES + 1, last[frame_id - FRAMES + 1 : frame_id + 1])
                    dataset.append(anno)
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

    def __init__(self, sequence_id, frame_id, matrixs):
        self.sequence_id = sequence_id
        self.frame_id = frame_id

        matrix1 = np.vstack([matrixs[0], [0,0,0,1]])
        matrix2 = np.vstack([matrixs[-1], [0,0,0,1]])

        rotation = np.matmul(np.linalg.inv(matrix2), matrix1)

        self.translation_mat = rotation[0:3,3]
        v = self.rotationMatrixToEulerAngles(rotation[:3,:3])
        self.matrix = np.array([self.translation_mat[0], self.translation_mat[1], self.translation_mat[2], v[0], v[1], v[2]])

    def get_matrix(self):
        return self.matrix

    def get_image(self):

        frame = None
        for i in range(self.frame_id, self.frame_id + FRAMES):
            camera1_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_0",  "{:06}.png".format(i))
            camera2_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence_id), "image_1",  "{:06}.png".format(i))

            camera1_image = cv2.imread(camera1_path, 0)
            camera2_image = cv2.imread(camera2_path, 0)

            camera1_image = camera1_image[:HEIGHT_ORIG, :WIDTH_ORIG]
            camera2_image = camera2_image[:HEIGHT_ORIG, :WIDTH_ORIG]

            camera1_image = cv2.resize(camera1_image, (WIDTH, HEIGHT))
            camera2_image = cv2.resize(camera2_image, (WIDTH, HEIGHT))

            curFrame = np.concatenate([np.expand_dims(camera1_image, axis=2), np.expand_dims(camera2_image, axis=2)], axis=2)
            if(frame is None):
                frame = curFrame
            else:
                frame = np.concatenate([frame, curFrame], axis = 2)

        frame = (frame-127.5)/127.5
        return frame

    def print_anno(self):
        print("#{0} #{1} {2}".format(self.sequence_id, self.frame_id, self.get_matrix()))



class TestLoader:
    def __init__(self, sequence):
        self.next = 0
        self.sequence = sequence

    def get_truth(self):
        path = os.path.join(LABELS_DIR, '{:02}.txt'.format(self.sequence))
        with open(path, "r") as file:
            dataset = []

            for line in file:
                numbers_text = line.split()
                numbers = np.zeros(len(numbers_text))
                for i in range(len(numbers_text)):
                    numbers[i] = float(numbers_text[i])

                projection_matrix = np.reshape(numbers,(MATRIX_ROWS, MATRIX_COLUMNS))
                dataset.append(projection_matrix)

        return dataset

    def get_test(self, batch_size):
        low = self.next
        high = self.next + batch_size
        #print(low,high)
        frame_id = low
        dataset = None

        while frame_id < high and os.path.exists(os.path.join(DATASET_DIR,  "{:02}".format(self.sequence), "image_0",  "{:06}.png".format(frame_id + 4))):
            frame = None
            for i in range(frame_id, frame_id + 5):
                camera1_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence), "image_0",  "{:06}.png".format(i))
                camera2_path = os.path.join(DATASET_DIR,  "{:02}".format(self.sequence), "image_1",  "{:06}.png".format(i))

                camera1_image = cv2.imread(camera1_path, 0)
                camera2_image = cv2.imread(camera2_path, 0)

                camera1_image = camera1_image[:HEIGHT_ORIG, :WIDTH_ORIG]
                camera2_image = camera2_image[:HEIGHT_ORIG, :WIDTH_ORIG]

                camera1_image = cv2.resize(camera1_image, (WIDTH, HEIGHT))
                camera2_image = cv2.resize(camera2_image, (WIDTH, HEIGHT))

                curFrame = np.concatenate([np.expand_dims(camera1_image, axis=2), np.expand_dims(camera2_image, axis=2)], axis=2)
                if(frame is None):
                    frame = curFrame
                else:
                    frame = np.concatenate([frame, curFrame], axis = 2)

            frame = (frame-127.5)/127.5

            if dataset is None:
                dataset = np.expand_dims(frame, axis = 0)
            else:
                dataset = np.concatenate((dataset, np.expand_dims(frame, axis = 0)))

            frame_id += 1
        self.next = high
        return dataset
