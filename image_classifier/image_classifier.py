from . import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import os
import cv2
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    # Number of features will be img_height * img_width* num_channels
    num_features = layer_shape[1:4].num_elements()
    # Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b.
    # Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


def show_progress(session, epoch, feed_dict_train, feed_dict_validate, val_loss, accuracy):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(data, path_name, num_iteration, batch_size, optimizer, cost, accuracy, x, y_true):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        start_iterations = 0
        saver = tf.train.Saver()

        for i in range(start_iterations, start_iterations + num_iteration):

            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

            feed_dict_tr = {x: x_batch, y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_tr)

            if i % int(data.train.num_examples / batch_size) == 0:
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(data.train.num_examples / batch_size))

                show_progress(session, epoch, feed_dict_tr, feed_dict_val, val_loss, accuracy)
                directory = dir_path + "/" + path_name + "-model"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saver.save(session, directory + "/" + path_name + "-model")

        start_iterations += num_iteration


def load_model(session, path_name):
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph(dir_path + '/' + path_name + '-model/' + path_name + '-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(session, tf.train.latest_checkpoint(dir_path + '/' + path_name + '-model/'))


def predict(image_path, path_name, classes):
    filename = image_path
    image_size = 128
    num_channels = 3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    with tf.Session() as session:
        # Let us restore the saved model
        load_model(session, path_name)
        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()
        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")
        # Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 2))
        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = session.run(y_pred, feed_dict=feed_dict_testing)
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        return dict(zip(classes, result.tolist()[0]))


class ImageClassifierCNN():
    def __init__(self, classes):
        seed(1)
        set_random_seed(2)

        self.batch_size = 32

        # Prepare input data
        self.classes = classes
        self.num_classes = len(classes)
        self.path_name = '-'.join(self.classes)

    def read_dataset(self):
        # We shall load all the training and validation images and labels into memory using openCV and use that during training
        self.data = dataset.read_train_sets(self.train_path, self.img_size, self.classes, validation_size=self.validation_size)

        print("Complete reading input data. Will Now print a snippet of it")
        print("Number of files in Training-set:\t{}".format(len(self.data.train.labels)))
        print("Number of files in Validation-set:\t{}".format(len(self.data.valid.labels)))

    def build_network_layer(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')

        # labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        y_true_cls = tf.argmax(self.y_true, axis=1)

        # Network graph params
        filter_size_conv1 = 3
        num_filters_conv1 = 32

        filter_size_conv2 = 3
        num_filters_conv2 = 32

        filter_size_conv3 = 3
        num_filters_conv3 = 64

        fc_layer_size = 128

        layer_conv1 = create_convolutional_layer(input=self.x, num_input_channels=self.num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)
        layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
        layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3)
        layer_flat = create_flatten_layer(layer_conv3)
        layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, use_relu=True)
        layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=self.num_classes, use_relu=False)

        y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

        y_pred_cls = tf.argmax(y_pred, axis=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=self.y_true)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        # 20% of the data will automatically be used for validation
        self.validation_size = 0.2
        self.img_size = 128
        self.num_channels = 3
        self.train_path = 'training_data'

        self.num_iteration = 3000

        self.read_dataset()
        self.build_network_layer()

        train(data=self.data, path_name=self.path_name, num_iteration=self.num_iteration, batch_size=self.batch_size, optimizer=self.optimizer, cost=self.cost, accuracy=self.accuracy, x=self.x, y_true=self.y_true)

    def predict(self, image_path):
        return predict(image_path=image_path, path_name=self.path_name, classes=self.classes)


if __name__ == '__main__':
    pass
    # image_classifier = ImageClassifier(classes=['dogs', 'cats'])
    # image_classifier.train()
    # result = image_classifier.predict(image_path=dir_path + '/' + sys.argv[1])
    # print(result)
