import os
import cv2 as cv

DATASET_DIR = os.path.join("dataset", "sequences")

class Resize:

    def resize(self, seq_dir):

        for seq in os.listdir(seq_dir):
            for cameras in os.listdir(seq):
                for img in os.listdir(cameras):
                    img_path = os.path.join(DATASET_DIR,  )
                    image = cv2.imread(img_path, 0)
                    height, width, channels = image.shape
#FIX ME!!!!                 cv.imwrite( ,cv.resize(camera1_image , (width1//2, height1//2)))


    pass


#resize all pict
Resize.resize(DATASET_DIR)