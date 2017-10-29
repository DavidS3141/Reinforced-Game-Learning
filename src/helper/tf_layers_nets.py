#!/user/bin/env python

'''tf_layers_nets.py: Serves functions returning layers and complete networks.'''

################################################################################

import tensorflow as tf
import numpy as np
from helper.tf_activations import get_activation

def fc_network(x, layer_sizes, name, act='relu'):
    with tf.name_scope(name):
        # get activation function
        act, weights_std_factor = get_activation(act)

        # create all variables for this network
        theta = []
        hiddens = []
        for i in range(len(layer_sizes)-2):
            hidden_curr, theta_curr = fc_layer(layer_sizes[i], layer_sizes[i+1], 'layer', act=act, weights_std_factor=weights_std_factor)
            theta += theta_curr
            hiddens.append(hidden_curr)
        output, theta_curr = fc_layer(layer_sizes[-2], layer_sizes[-1], 'output', act=None)
        theta += theta_curr
        hiddens.append(output)

        # apply network on input
        for i in range(len(hiddens)):
            x = hiddens[i](x)
        return x

def fc_layer(input_dim, output_dim, layer_name, act=tf.nn.relu, weights_std_factor=2.):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses act to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name+'_vars'):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim],weights_std_factor)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
    def layer(input_tensor):
        with tf.name_scope(layer_name):
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate, collections=['v2'])
            if act is None:
                return preactivate
            else:
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations, collections=['v2'])
                return activations
    return layer, [weights, biases]

def weight_variable(dims, weights_std_factor=2.):
    return tf.Variable(xavier_init(dims, weights_std_factor))

def bias_variable(dims):
    return tf.Variable(tf.zeros(shape=dims))

def xavier_init(size, weights_std_factor=2.):
    in_dim = size[0]
    xavier_stddev = tf.sqrt(weights_std_factor/in_dim)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=['v2'])
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, collections=['v2'])
        tf.summary.scalar('max', tf.reduce_max(var), collections=['v2'])
        tf.summary.scalar('min', tf.reduce_min(var), collections=['v2'])
        tf.summary.histogram('histogram', var, collections=['v2'])
