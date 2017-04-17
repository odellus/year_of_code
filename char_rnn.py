#! /usr/bin/env python3

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
import string
rnn = tf.contrib.rnn

# def flatten(tensor, keep):
#     fixed_shape = tensor.get_shape().as_list()
#     start = len(fixed_shape) - keep
#     left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
#     out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
#     flat = tf.reshape(tensor, out_shape)
#     return flat

char_dict = {x:k for k,x in enumerate(string.printable)}
# Get file from https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
fname = "/home/thomas/downloads/t8.shakespeare.txt"
with open(fname, "r") as f:
    data = f.read()

D = len(char_dict)
N = len(data)
n_hidden = 128
n_steps = 200
n_batch = 25

max_epoch = 200
steps_per_epoch = 500

u = np.zeros((N, D))
for k,x in enumerate(data):
    u[k,char_dict[x]] = 1.

n_seqs = int(N / n_steps)
n_rem = N % n_steps
# print(n_seqs, n_rem, N-n_seqs*n_steps)
u = u[:-n_rem,:].reshape(n_steps,n_seqs,D)

# [timestep, mini-batch, feature_dims]
# or [n_steps, n_batch, D]
x = tf.placeholder(tf.float32, [None, None, D])
y = tf.placeholder(tf.float32, [None, None, D])

# index = tf.placeholder(tf.int32, [None,])

initializer = tf.random_uniform_initializer(-1.,1.)
cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
cell_out = rnn.OutputProjectionWrapper(cell, D)
outputs, _ = dynamic_rnn(cell_out, x, dtype=tf.float32, time_major=True)
pred = tf.argmax(tf.nn.softmax(outputs),axis=-1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
print(outputs.shape)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(max_epoch):
        for k in range(steps_per_epoch):
            # Grab a random sample of data.
            batch_indices = np.random.choice(n_seqs, n_batch)
            batch_xs = u[:-1,batch_indices,:]
            batch_ys = u[1:,batch_indices,:]

            # Make a feed dict to push to the session.
            feed_dict = {x:batch_xs, y:batch_ys}

            # Update the parameters with Adam Gradient Descent optimization.
            sess.run(optimizer, feed_dict=feed_dict)

            # Now we have to check the loss function to see how we're doing.
            fetches = [loss]
            [l] = sess.run(fetches, feed_dict=feed_dict)
            print("Now on step {0} with a loss of {1}".format(epoch*steps_per_epoch+k, l))
            if l < 1e-4:
                print("Training converged to less than 1e-4, so we've stopped.")
                saver.save(sess, "./model.ckpt")
                break
        saver.save(sess, "./model.ckpt")
