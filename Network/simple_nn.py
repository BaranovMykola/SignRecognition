import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


class TF_Model:
    def __init__(self, classes_count):
        self.classes_count = classes_count

        self.model = tf.estimator.Estimator(self.model_fn, model_dir='./Tensorboard/')
        


    def neural_net(self, x):
        x = x
        cnn1 = tf.layers.conv2d(inputs=x, filters=8, activation=tf.nn.relu, kernel_size=(3,3))
        mp1 = tf.layers.max_pooling2d(inputs=cnn1, pool_size=(2,2), strides=2)
        cnn2 = tf.layers.conv2d(inputs=mp1, filters=8, activation=tf.nn.relu, kernel_size=(3,3))
        mp2 = tf.layers.max_pooling2d(inputs=cnn2, pool_size=(2, 2), strides=2)
        cnn3 = tf.layers.conv2d(inputs=mp2, filters=8, activation=tf.nn.relu, kernel_size=(3,3))
        mp3 = tf.layers.max_pooling2d(inputs=cnn3, pool_size=(2, 2), strides=2)

        features = tf.reshape(mp3, ([-1,6*6*8]))

        layer_1 = tf.layers.dense(features, 6*6*8)
        layer_3 = tf.layers.dense(layer_1, 64)
        out_layer = tf.layers.dense(layer_3, self.classes_count)
        return out_layer


# Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network


        logits = self.neural_net(features)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels= tf.cast(labels, dtype=tf.int32)))

        logging_hook = tf.train.LoggingTensorHook({"loss": loss_op}, every_n_iter=10)

        # loss_op = tf.losses.softmax_cross_entropy(tf.cast(labels, dtype=tf.int32), logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op},
            training_hooks=[logging_hook])

        return estim_specs


    def train(self, input_fn):
        tf.logging.set_verbosity(tf.logging.INFO)
        # self.model.model_dir = "./"
        self.model.train(input_fn, steps=100)



    def test(self, input_fn):
        res = self.model.evaluate(input_fn, steps=100)

        print("Testing Accuracy:", res['accuracy'], res)
        
    
    def predict(self, np_tensor):
        input = tf.estimator.inputs.numpy_input_fn(np_tensor, shuffle=False)
        predict = self.model.predict(input)
        predictes = list([x for x in predict])
        return predict

    # def my_input_fn(self):