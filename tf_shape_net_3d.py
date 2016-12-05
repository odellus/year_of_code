#! /usr/bin/env python3
# -*- coding: utf-8

# Author : Thomas Wood, thomas@synpon.com
# File   : tf_shape_net_3d.py
# Descr  : A TensorFlow implementation of the network represented in
#          https://arxiv.org/abs/1406.5670

import tensorflow as tf

def conv3d(input_, kh, kw, kd, output_cdim, stride=2, scope=None, with_w=False):
    input_cdim = input_.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("weights",
            shape=[kd, kh, kw, input_cdim, output_cdim],
            initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias",
            shape=[output_cdim],
            initializer=tf.constant_initializer(0.0))
    return tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding="VALID") + b

def batch_norm(input_, scope=None):
    # Input must be a 5D Tensor.
    in_cdim, h, w, d, out_cdim = input_.get_shape().as_list()
    norm_shape = [1, 1, 1, 1, out_cdim]
    var_eps = 1e-6
    with tf.variable_scope(scope):
        offset = tf.get_variable("offset",
            shape=norm_shape,
            initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable("scale",
            shape=norm_shape,
            initializer=tf.random_normal_initializer(stddev=0.1))
    mu, var = tf.nn.moments(input_, [0,1,2,3],keep_dims=True)
    return tf.nn.batch_normalization(input_, mu, var, offset, scale, var_eps, name=scope)

def linear(input_, output_size, scope=None):
    # We assume the input_ has been flattened using
    # tf.reshape(input_, [self.batch_size, -1])
    input_dim = input_.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("weights",
            shape=[input_dim, output_size],
            initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias",
            shape=[output_size],
            initializer=tf.constant_initializer(0.0))
    return tf.matmul(input_, w) + b

class ShapeNet:
    def __init__(self, session, batch_size, z_dim, n_logits):
        self.sess = session
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.n_logits = n_logits
        self.build_model(session)

    def build_model(self, session):

        # Batch of 30 X 30 X 30 input shapes or 27000 boolean voxels.
        x = tf.placeholder(tf.float32, shape=[self.batch_size, 30, 30, 30, 1], name="input")
        x = batch_norm(x, scope="bn1")

        # After normalizing the input over the batch do the first convolution.
        x = tf.nn.elu(conv3d(x, 6, 6, 6, 48, stride=2, scope="conv1"))
        x = batch_norm(x, scope="bn2")

        x = tf.nn.elu(conv3d(x, 5, 5, 5, 160, stride=2, scope="conv2"))
        x = batch_norm(x, scope="bn3")

        x = tf.nn.elu(conv3d(x, 4, 4, 4, 512, stride=1, scope="conv3"))
        x = batch_norm(x, scope="bn4")

        x = tf.reshape(x, [self.batch_size, -1])

        x = tf.nn.relu(linear(x, self.z_dim, scope="fc1"))

        self.logits = tf.nn.softmax(linear(x, self.n_logits, scope="logits"))

        self.t_vars =tf.trainable_variables()
