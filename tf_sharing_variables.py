#! /usr/bin/env/ python
# -*- coding: utf-8

import tensorflow as tf
import numpy as np


def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1,1,1,1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)

"""
tf.get_variable(<name>, <shape>, <initializer>): Creates or returns a variable with a given name.

tf.variable_scope(<scope_name>): Manages namespaces for names passed to tf.get_variable()

tf.get_variable() is used to get or create a variable instead of a direct call to tf.Variable().
It uses and initializer instead of passing the values directly. An initializer is a function that takes the
shape and provides the tensor with that shape. Here are some initializers available in TensorFlow.

tf.constant_initializer(value) initalizes everything to the provided value.
tf.random_uniform_initalizer(a, b) initializes uniformly from [a,b]
tf.random_normal_initalizer(mean, stddev) initializes from normal distribution with given mean and stddev.
"""

def conv_relu(input_images, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_images, weights, strides=[1,1,1,1], padding='SAME')
    return tf.nn.relu(conv + biases)

def another_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5,5,32,32],[32])
    with tf.variable_scope("conv2"):
        relu2 = conv_relu(relu1, [5,5,32,32],[32])
    return relu2

def try_it_out():
