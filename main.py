from DatasetProcessing import dataset_reader
from DatasetProcessing import vector_image_dataset_reader
from Network import simple_nn
import numpy as np
import random
import tensorflow as tf
import cv2
import functools

TRAIN_FOLDER = './Data/Train/'

random.seed(0)
tf.set_random_seed(1)


if __name__ == '__main__':
    dsReader = vector_image_dataset_reader.VectorImageDatasetReader(TRAIN_FOLDER, (64,64))

    print('Building model...')
    tf.InteractiveSession()
    model = simple_nn.TF_Model(43)
    print('Model builded. Start training...')

    ds = dsReader.build_dataset(1,1)
    model.train(ds)
    print('Model trained. Testing...')
    # model.test(dsReader.build_dataset(128,1))
    #
    # img = cv2.imread("2.png")
    # img = cv2.resize(img, (64,64))
    # vector_size = functools.reduce(lambda x, y: x * y, img.shape)
    # img = img.astype(np.float32)
    # img = img/255.0
    # img = img.reshape(vector_size)
    #
    #
    # model.predict(np.array([img]))