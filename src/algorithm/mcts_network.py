#!/usr/bin/env python
'''
algorithm/mcts_network.py: Implement a network class used by Monte-Carlo tree
    search.
'''
###############################################################################
from helper.general import create_batch_generator
from helper.tf_network import fc_network

import numpy as np
import tensorflow as tf


class MCTS_Network(object):
    def __init__(self, in_dim, priors_dim, values_dim, batch_size=1,
                 layer_sizes=[512, 256, 256, 256, 256]):
        self.batch_size = batch_size
        self.net_in = tf.placeholder(tf.float32, shape=(None, in_dim))
        network = fc_network([in_dim] + layer_sizes + [priors_dim
                             + values_dim], 'FFN', 'relu')
        logits = network(self.net_in)
        self.priors_logits = logits[:, :priors_dim]
        self.values_logits = logits[:, priors_dim:]
        self.priors_out = tf.nn.softmax(self.priors_logits)
        self.values_out = tf.nn.softmax(self.values_logits)
        self.priors_target = tf.placeholder(tf.float32,
                                            shape=(None, priors_dim))
        self.values_target = tf.placeholder(tf.float32,
                                            shape=(None, values_dim))
        priors_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.priors_target, logits=self.priors_logits)
        priors_loss = tf.reduce_mean(priors_loss)
        values_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.values_target, logits=self.values_logits)
        values_loss = tf.reduce_mean(values_loss)

        theta = tf.get_collection(tf.GraphKeys.WEIGHTS)
        print('Trainable weights:')
        print(theta)

        self.regularization = tf.contrib.layers.apply_regularization(
            tf.nn.l2_loss, weights_list=theta)
        self.loss = priors_loss + values_loss
        self.regularized_loss = self.loss + self.regularization / 3000.0
        self.optimize = tf.train.AdamOptimizer().minimize(
            self.regularized_loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def evaluate(self, in_state):
        priors, values = self.session.run(
            [self.priors_out, self.values_out],
            feed_dict={self.net_in: [in_state]})
        return priors[0], values[0]

    def train(self, state_in, priors_target, values_target):
        state_in = np.array(state_in, dtype=np.float32)
        priors_target = np.array(priors_target, dtype=np.float32)
        values_target = np.array(values_target, dtype=np.float32)
        batch_generator = create_batch_generator(
            self.batch_size, [state_in, priors_target, values_target])
        epoch_flt = 0.0
        while epoch_flt < 1.0:
            state_in_curr, priors_target_curr, values_target_curr, epoch_flt \
                = next(batch_generator)
            _, loss, regul = self.session.run(
                [self.optimize, self.loss, self.regularization],
                feed_dict={self.net_in: state_in_curr,
                           self.priors_target: priors_target_curr,
                           self.values_target: values_target_curr}
            )
            print('loss: %e, regul: %e' % (loss, regul))
