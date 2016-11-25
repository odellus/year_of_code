#! /usr/bin/env python
# -*- coding: utf-8

import tensorflow as tf
import gym
import numpy as np

# Define a TensorFlow model that takes in an image and outputs an action
# to be fed to an OpenAI Gym environment.

def conv_relu(input_images, kernel_shape, bias_shape, stride=1):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_images, weights, strides=[1,stride,stride,1], padding='VALID')
    return tf.nn.relu(conv + biases)

def fc_relu(flat_input, nin, nout):
    weights = tf.get_variable("weights", [nin,nout],
        initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", [nout],
        initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.matmul(flat_input, weights) + biases)

def image_filter(input_images):
    nin = tf.shape(input_images)[-1]
    with tf.variable_scope("conv1"):
        relu1 = conv_relu(input_images, [5,5,nin,24], [24])
    with tf.variable_scope("conv2"):
        relu2 = conv_relu(relu1, [5,5,24,32], [32])
    return relu2


class Agent:
    def __init__(self, environment, policy):
        self.env = gym.make(environment)
        self.init_img = self.env.render(mode='rgb_array')
        self.n_action = self.env.action_space.n
        self.iw, self.ih, self.nin = self.init_img.shape

    def rollout(self, policy, n_episodes, n_timesteps):
        pass



class Policy:
    def __init__(self, n_actions, iw, ih, nin):
        self.n_actions = n_actions
        self.img_no = tf.placeholder(tf.float32, shape=[None, iw, ih, nin])
        # self.a_n = tf.placeholder(tf.int32, shape=[None])
        # self.q_n = tf.placeholder(tf.float32, shape=[None])
        # self.oldpdist_np = tf.placeholder(tf.float32, shape=[None, n_actions])
        self.keep_prob = tf.placeholder(tf.float32)
        # self.lam = tf.placeholder(tf.float32)
        self.n_batch = tf.shape(self.img_no)[0]

        with tf.variable_scope("conv1"):
            relu1 = conv_relu(self.img_no, [5,5,nin,24],[24], stride=2)

        with tf.variable_scope("conv2"):
            relu2 = conv_relu(relu1, [5,5,24,36], [36], stride=2)

        with tf.variable_scope("conv3"):
            relu3 = conv_relu(relu2, [3,3,36,64], [64], stride=2)

        with tf.variable_scope("conv4"):
            relu4 = conv_relu(relu3, [5,5,64,64], [64], stride=2)

        with tf.variable_scope("avgpool1"):
            avgpool1 = tf.nn.avg_pool(relu4, [1, 5, 5, 1], strides=[1,1,1,1], padding='VALID')

        avgpool1_shape = avgpool1.get_shape().as_list()
        avgpool1_flat_n = np.prod(avgpool1_shape[1:])
        avgpool1_flat = tf.reshape(avgpool1, [self.n_batch, avgpool1_flat_n])

        with tf.variable_scope("fc1"):
            fc1 = fc_relu(avgpool1_flat, avgpool1_flat_n, 1164)
            fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)
        with tf.variable_scope("fc2"):
            fc2 = fc_relu(fc1_dropout, 1164, 512)
            fc2_dropout = tf.nn.dropout(fc2, self.keep_prob)
        with tf.variable_scope("fc3"):
            fc3 = fc_relu(fc2_dropout, 512, 128)
            fc3_dropout = tf.nn.dropout(fc3, self.keep_prob)
        with tf.variable_scope("fc4"):
            fc4 = fc_relu(fc3_dropout, 128, 64)
            fc4_dropout = tf.nn.dropout(fc4, self.keep_prob)
        with tf.variable_scope("fc5"):
            weights = tf.get_variable("weights", [64, n_actions],
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable("biases", [n_actions],
                initializer=tf.constant_initializer(0.0))
            fc5 = tf.nn.softmax(tf.matmul(fc4_dropout, weights) + biases)


        self.probs_na = fc5
        self.pred_action = tf.argmax(self.probs_na, 1)
        # logprobs_na = tf.log(self.probs_na)
        # idx_flattened = tf.range(0,self.n_batch) * n_actions + self.a_n
        #
        # logps_n = tf.gather(tf.reshape(logprobs_na, [-1]), idx_flattened)
        #
        # self.surr = tf.reduce_mean(tf.mul(logps_n, self.q_n))
        #
        # params = tf.trainable_variables()
        #
        # self.surr_grads = tf.gradients(self.surr, params)
        #
        # self.kl = tf.reduce_mean(
        #     tf.reduce_sum(
        #         tf.mul(self.oldpdist_np, tf.log(tf.div(self.oldpdist_np, self.probs_na))), 1
        #     )
        # )
        # penobj = tf.sub(self.surr, tf.mul(self.lam, self.kl))

    # def decision(self, session, image):
    #     feed_dict = {self.img_no:image}
    #     self.current_action = session.run([self.pred_action],feed_dict=feed_dict)


def test_Policy():

    # Start an interactive session.
    session = tf.InteractiveSession()

    # Initialize policy.
    policy = Policy(18, 210, 160, 3)

    # Initialize variables in policy.
    session.run(tf.global_variables_initializer())
    
    env = gym.make('Boxing-ram-v0')
    img = env.render(mode='rgb_array')
    feed_dict = {policy.img_no:img.reshape(1,210,160,3),
        policy.keep_prob:1.0}
    action = session.run([policy.pred_action],feed_dict=feed_dict)
    print(action)


if __name__ == "__main__":
    test_Policy()
