import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import os
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


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    global session, accuracy
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(session, num_iteration, x, y_true):
    global optimizer, cost
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

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            directory = dir_path + "/dogs-cats-model"
            if not os.path.exists(directory):
                os.makedirs(directory)
            saver.save(session, directory + "/dogs-cats-model")

    start_iterations += num_iteration


if __name__ == '__main__':
    seed(1)
    set_random_seed(2)

    batch_size = 32

    # Prepare input data
    classes = ['dogs', 'cats']
    num_classes = len(classes)

    # 20% of the data will automatically be used for validation
    validation_size = 0.2
    img_size = 128
    num_channels = 3
    train_path = 'training_data'

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

    # labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    # Network graph params
    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64

    fc_layer_size = 128

    layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
    layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3)
    layer_flat = create_flatten_layer(layer_conv3)
    layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, use_relu=True)
    layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train(session=session, num_iteration=3000, x=x, y_true=y_true)
