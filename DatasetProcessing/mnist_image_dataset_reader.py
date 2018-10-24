import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


from DatasetProcessing import base_image_dataset_reader
import tensorflow as tf
import functools

class MnistDatasetReader(base_image_dataset_reader.BaseDatasetReader):
    def __init__(self, train_dir, image_size):
        base_image_dataset_reader.BaseDatasetReader.__init__(self, train_dir)
        self.image_size = image_size


    def parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        return image, label


    def train_preprocess(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, self.image_size)
        return image, label