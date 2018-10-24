from DatasetProcessing import dataset_reader
import tensorflow as tf

class BaseDatasetReader():
    def __init__(self, train_dir):
        self.train_dir = train_dir


    def parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        return image, label


    def train_preprocess(self, image, label):
        return image, label


    def build_dataset(self, batch_size, epochs, prefetch_count = 1):
        def build_dataset_lambda():
            files, labels = dataset_reader.read_all_from_directory(self.train_dir)
            dataset = tf.data.Dataset.from_tensor_slices((files, labels))
            dataset = dataset.shuffle(len(files))
            dataset = dataset.repeat(epochs)
            dataset = dataset.map(self.parse_function, num_parallel_calls=4)
            dataset = dataset.map(self.train_preprocess, num_parallel_calls=4)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(prefetch_count)
            return dataset

        return build_dataset_lambda