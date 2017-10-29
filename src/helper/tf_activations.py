#!/user/bin/env python

'''tf_activations.py: Implement different activation functions for the network.'''

################################################################################

import tensorflow as tf

def leaky_relu(x, alpha=0.2, name='leaky relu'):
    with tf.name_scope(name):
        return tf.maximum(x, x*alpha)

def selu(x, name='selu'):
    with tf.name_scope(name):
        return 1.7580993408473768599402175208123 * (tf.exp(tf.minimum(x,tf.zeros_like(x))) - tf.ones_like(x)) + 1.0507009873554804934193349852946 * tf.nn.relu(x)

def sigmoid_num(x, name='sigmoid num'):
    with tf.name_scope(name):
        return 2.8*tf.nn.sigmoid(x)

def softsign_num(x, name='softsign num'):
    with tf.name_scope(name):
        return 2*tf.nn.softsign(x)

def tanh_num(x, name='tanh num'):
    with tf.name_scope(name):
        return 1.5*tf.nn.tanh(x)

def get_activation(name):
    if name == 'elu':
        return tf.nn.elu, 1.
    elif name == 'elu_num':
        return tf.nn.elu, 1.6444
    elif name == 'leaky_relu':
        return leaky_relu, 1.92308
    elif name == 'relu':
        return tf.nn.relu, 2.
    elif name == 'relu6':
        return tf.nn.relu6, 2.001745
    elif name == 'selu':
        return selu, 1.
    elif name == 'sigmoid':
        return tf.nn.sigmoid, 12.8
    elif name == 'sigmoid_num':
        return sigmoid_num, 2.55805
    elif name == 'softplus':
        return tf.nn.softplus, 2.701875684
    elif name == 'softplus_num':
        return tf.nn.softplus, 1.66423
    elif name == 'softsign':
        return tf.nn.softsign, 1.
    elif name == 'softsign_num':
        return softsign_num, 2.02481
    elif name == 'tanh':
        return tf.nn.tanh, 1.
    elif name == 'tanh_num':
        return tanh_num, 1.31510
    else:
        return None, None
