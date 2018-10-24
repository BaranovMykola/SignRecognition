from DatasetProcessing import base_image_dataset_reader
import functools
import tensorflow as tf

class VectorImageDatasetReader(base_image_dataset_reader.BaseDatasetReader):
    def __init__(self, train_dir, image_size):
        base_image_dataset_reader.BaseDatasetReader.__init__(self, train_dir)
        self.image_size = image_size


    def train_preprocess(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, self.image_size)
        vector_size = functools.reduce(lambda x,y: x*y, image.shape)
        feature_vector = tf.reshape(image, shape=[vector_size])
        return feature_vector, label

