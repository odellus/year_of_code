#! /usr/bin/env python3

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
import string

char_dict = {x:k for k,x in enumerate(string.printable)}


def get_shakespeare_training_set(n_steps=200):
    # Get file from https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
    fname = "/home/thomas/downloads/t8.shakespeare.txt"
    with open(fname, "r") as f:
        data = f.read()
    D = len(char_dict)
    N = len(data)
    u = np.zeros((N, D))
    for k,x in enumerate(data):
        u[k,char_dict[x]] = 1.

    n_seqs = int(N / n_steps)
    n_rem = N % n_steps
    # print(n_seqs, n_rem, N-n_seqs*n_steps)
    return u[:-n_rem,:].reshape(n_steps,n_seqs,D)


class Model(object):
    def __init__(self, n_hidden=128, n_steps=200, n_batch=250, n_out=100, fname='./model.ckpt'):
        self.n_hidden = n_hidden
        self.n_batch = n_batch
        self.n_steps = n_steps
        self.n_out = n_out
        self.learning_rate = 0.0001
        self._vars_initialized = False
        self.range = [-0.3,0.3]
        self.fname=fname

        lbound, rbound = self.range
        # [timestep, mini-batch, feature_dims]
        # or [n_steps, n_batch, D]
        self.x = tf.placeholder(tf.float32, [None, None, n_out])
        self.y = tf.placeholder(tf.float32, [None, None, n_out])
        self.initializer = tf.random_uniform_initializer(lbound, rbound)
        cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=self.initializer)
        self.cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, n_out)
        self.sess = tf.Session()
        # self.saver = tf.train.Saver()

    def step(self, _input, state):
        # assert type(state) == tf.contrib.rnn.LSTMStateTuple
        return self.cell_out(_input, state)

    def restore(self):
        try:
            self.saver.restore(self.sess, self.fname)
        except:
            print("Could not load the checkpoint file.")
            raise

    def save(self):
        self.saver.save(self.sess, self.fname)

    def set_checkpoint_fname(self, fname):
        self.fname = fname

    def _initialize_vars(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self._vars_initialized = True


    def train(self, dataset, max_epoch=1000, steps_per_epoch=1000):
        n_batch, n_seqs, n_out = dataset.shape
        # define a loss function and use the dynamic_rnn fxn.
        outputs, _ = dynamic_rnn(self.cell_out, self.x, dtype=tf.float32, time_major=True)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs, labels=tf.argmax(self.y, axis=-1)
                )
            )
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
            ).minimize(self.loss)

        if not self._vars_initialized:
            self._initialize_vars()
        self.saver = tf.train.Saver()
        old_loss = 0.
        converge_tol = 1e-8
        for epoch in range(max_epoch):
            for k in range(steps_per_epoch):
                i = epoch*steps_per_epoch + k
                batch_inds = np.random.choice(n_seqs, n_batch)
                batch_xs = dataset[:-1, batch_inds, :]
                batch_ys = dataset[1:, batch_inds, :]

                feed_dict = {self.x:batch_xs, self.y:batch_ys}
                self.sess.run(self.optimizer, feed_dict=feed_dict)
            fetches = [self.loss]
            [ loss ] = self.sess.run(fetches, feed_dict=feed_dict)
            if np.abs(loss - old_loss) < converge_tol:
                print("Training converged to a value of {0}.".format(loss))
                self.save()
                break
            old_loss = loss
            print("Now on the {0}-th epoch with loss {1}.".format(epoch, loss))
            self.save()

    def generative_model(self, n_steps=2000):
        self.init_char = tf.placeholder(tf.float32, [None, self.n_out])
        self.init_c = tf.placeholder(tf.float32, [None, self.n_hidden])
        self.init_h = tf.placeholder(tf.float32, [None, self.n_hidden])

        init_state = tf.contrib.rnn.LSTMStateTuple(self.init_c,self.init_h)

        chars = []
        for k in range(n_steps):
            if k == 0:
                out_vec, new_state = self.step(self.init_char, init_state)
                chars.append(out_vec)
            else:
                out_vec, new_state = self.step(out_vec, new_state)
                chars.append(out_vec)
        self.state = new_state
        self.chars = tf.stack(chars,axis=1)

    def generate_text(self, n_steps=100, fname=None):
        self.generative_model(n_steps=n_steps)
        if fname is not None:
            self.set_checkpoint_fname(fname)
        if not self._vars_initialized:
            self._initialize_vars()
        self.restore()

        init_char = np.zeros((1, self.n_out))
        init_char[0, char_dict[' ']] = 1.
        init_c = 2*np.random.rand(1, self.n_hidden)-1.
        init_h = 2*np.random.rand(1, self.n_hidden)-1.
        fetches = [ self.chars ]
        feed_dict = {self.init_c:init_c, self.init_h:init_h, self.init_char:init_char}
        chars = self.sess.run(fetches, feed_dict=feed_dict)
        char_labels = np.argmax(chars,axis=-1)
        txt = []
        for k in range(char_labels.shape[-1]):
            txt.append(string.printable[char_labels[0,0,k]])
        txt = ''.join(txt)
        print(txt)


def train_and_generate(n_steps=400, gen_steps=2000, fname=None):
    with tf.variable_scope("lstm_weights") as scope:
        training_data = get_shakespeare_training_set(n_steps=n_steps)
        m = Model(n_steps=n_steps, fname=fname)
        m.train(training_data,max_epoch=200, steps_per_epoch=50)
    with tf.variable_scope("lstm_weights/rnn") as scope:
        scope.reuse_variables()
        m.generate_text()


if __name__ == "__main__":
    train_and_generate(fname='./do_this_thang.ckpt')
