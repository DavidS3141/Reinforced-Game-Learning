'''
helper/tf_network.py: Some helper functions to create a simple TensorFlow
    network.
'''
###############################################################################
from helper.tf_activations import get_activation

import tensorflow as tf


def fc_network(layer_sizes, name, act_name='relu'):
    with tf.name_scope(name + '_vars'):
        # get activation function
        act, pre_std, post_mean = get_activation(act_name)
        if act is None:
            raise Exception(
                'You have to specify a valid discriminator activation '
                'function!\n%s is not a valid activation function!' % act_name)

        # create all variables for this network
        layer_vars = []
        layer_funcs = []
        in_mean = 0.
        in_std = 1.
        for i in range(len(layer_sizes) - 2):
            layer_func, layer_var = fc_layer(
                layer_sizes[i], layer_sizes[i + 1], 'layer', act=act,
                target_std=pre_std, in_mean=in_mean, in_std=in_std)
            in_mean = post_mean
            layer_vars += layer_var
            layer_funcs.append(layer_func)
        output, layer_var = fc_layer(
            layer_sizes[-2], layer_sizes[-1], 'output', act=None,
            target_std=1., in_mean=in_mean, in_std=in_std)
        layer_vars += layer_var
        layer_funcs.append(output)

    # apply network on input
    def network(x):
        with tf.name_scope(name):
            for i in range(len(layer_funcs)):
                x = layer_funcs[i](x)
            return x

    return network


def xavier_init(size, weights_std=2.):
    in_dim = size[0]
    xavier_stddev = tf.sqrt(weights_std / in_dim)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def variable_summaries(var):
    '''
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=['v2'])
        with tf.name_scope('std'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('std', stddev, collections=['v2'])
        tf.summary.scalar('max', tf.reduce_max(var), collections=['v2'])
        tf.summary.scalar('min', tf.reduce_min(var), collections=['v2'])
        tf.summary.histogram('histogram', var, collections=['v2'])


def fc_layer(input_dim, output_dim, layer_name, act=tf.nn.relu,
             target_std=1.7128, in_mean=0.6833, in_std=1.):
    '''
    Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    '''
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name + '_vars'):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(
                [input_dim, output_dim], target_std**2 / (in_mean**2
                                                          + in_std**2))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)

    def layer(input_tensor):
        with tf.name_scope(layer_name):
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram(
                    'pre_activations', preactivate, collections=['v2'])
            if act is None:
                return preactivate
            else:
                activations = act(preactivate, name='activation')
                tf.summary.histogram(
                    'activations', activations, collections=['v2'])
                tf.summary.scalar('activations/std',
                                  tf.sqrt(tf.reduce_mean(tf.square(
                                      activations
                                      - tf.reduce_mean(activations)))),
                                  collections=['v2', 'watch'])
                return activations
    return layer, [weights, biases]


def weight_variable(dims, weights_std=2.):
    return tf.Variable(xavier_init(dims, weights_std),
                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                    tf.GraphKeys.WEIGHTS])


def bias_variable(dims):
    return tf.Variable(tf.zeros(shape=dims),
                       collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                    tf.GraphKeys.BIASES])
