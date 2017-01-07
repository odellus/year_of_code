"""
File: tf_shape_completion.py

Author: Thomas Wood, thomas@synpon.com

Description: The same network used to evaluate 3D volumes of voxel as the paper
             https://arxiv.org/abs/1609.08546.
"""

import tensorflow as tf

def conv3d(input_, kh, kw, kd, output_cdim, stride=1, scope=None, with_w=False):
    input_cdim = input_.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("weights",
            shape=[kd, kh, kw, input_cdim, output_cdim],
            initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias",
            shape=[output_cdim],
            initializer=tf.constant_initializer(0.0))
    return tf.nn.conv3d(input_, w, [1, stride, stride, stride, 1], padding="VALID") + b

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

def maxpool3d(input_, kd, kh, kw, stride=1,scope=None):
    input_cdim = input_.get_shape().as_list()[-1]
    batch_size = input_.get_shape().as_list()[0]
    kernel = [batch_size, kd, kh, kw, input_cdim]
    return tf.nn.max_pool3d(input_, kernel,
        strides=[1,stride,stride,stride,1], padding="VALID", name=scope)

class ShapeNet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.x_ = tf.placeholder(tf.float32, shape=[self.batch_size, 40, 40, 40, 1], name="input")

        # Commenting out the target and loss functions. This is just a forward
        # model.

        # self.y_ = tf.placeholder(tf.float32, shape=[self.batch_size, 40,40,40,1], name="target")

        self.build_model()
        # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.y, self.y_)

    def build_model(self):
        """
        Just builds a topological model.
        """
        # Bring the input placeholder in to local variable from the class.
        x = self.x_

        # Perform 2 3D convolution without max pooling.
        x = tf.nn.relu(conv3d(x, 4, 4, 4, 64, scope="conv_1"))
        x = tf.nn.relu(conv3d(x, 4, 4, 4, 64, scope="conv_2"))

        # Downsample with max pooling.
        x = maxpool3d(x, 2, 2, 2, scope="max_pool_1")

        # Do another convolution.
        x = tf.nn.relu(conv3d(x, 4, 4, 4, 64, scope="conv_3"))

        # Max pooling one more time.
        x = maxpool3d(x, 2, 2, 2, scope="max_pool_2")

        # Flatten for use in fully connected layers.
        x = tf.reshape(x, [self.batch_size, -1])

        # Intermediate fc layer.
        x = tf.nn.relu(linear(x, 5000, scope="fc_1"))

        # Expand back out to a flat 40**3 vector with another fc layers.
        x = tf.nn.sigmoid(linear(x, 40**3, scope="fc_2"))

        # Reshape back to batch_size X 40 X 40 X 40 X 1

        x = tf.reshape(x, [self.batch_size, 40, 40, 40, 1])

        self.y = x
        self.t_vars = tf.trainable_variables()
