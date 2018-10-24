from DatasetProcessing import dataset_reader
from DatasetProcessing import vector_image_dataset_reader
from DatasetProcessing import image_dataset_reader
from Network import simple_nn
import numpy as np
import random
import tensorflow as tf
import cv2
import functools
import sys
import os

TRAIN_FOLDER = './Data/Train/'

random.seed(0)
tf.set_random_seed(1)


if __name__ == '__main__':
    dsReader = image_dataset_reader.ImageDatasetReader(TRAIN_FOLDER, (64,64))

    act = 'Test'

    if len(sys.argv) > 1:
        act = sys.argv[1]



    print('Building model...')
    tf.InteractiveSession()
    model = simple_nn.TF_Model(43)
    print('Model builded. Start training...')

    if act=='Test':
        # ipt = []
        # for i in range(10):
        #     img = cv2.imread(str(i)+".png")
        #     img = cv2.resize(img, (64,64))
        #     # vector_size = functools.reduce(lambda x, y: x * y, img.shape)
        #     img = img.astype(np.float32)
        #     img = img/255.0
        #     # img = img.reshape(vector_size)
        #     ipt.append(img)
        #
        #
        # p = model.predict(np.array(ipt))
        # for i in range(10):
        #     print(i, p[i])
        dir = './Data/Test/Images/'
        idx = 0
        ipt = []
        names = []
        with open('ann.txt', 'w+') as fl:
            for f in os.listdir(dir):
                name = os.path.join(dir, f)
                img = cv2.imread(name)
                _img = img
                img = cv2.resize(img, (64,64))
                img = img.astype(np.float32)
                img = img/255.0
                idx += 1
                ipt.append(img)
                names.append(name)

                if idx%10 == 0:
                    p = model.predict(np.array(ipt))

                    for i in range(idx):
                        fl.write('{0} {1}\n'.format(names[i], p[i]))
                    idx = 0
                    ipt.clear()
                    names.clear()


    elif act=='Train':
        ds = dsReader.build_dataset(8,10)
        model.train(ds)
        print('Model trained. Testing...')
        model.test(dsReader.build_dataset(8,1))

    else:
        print('Unknown act <{0}>'.format(act))