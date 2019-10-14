#!/usr/bin/env python

'''inference.py: Implement the network in tensorflow to make predictions.'''

################################################################################

import tensorflow as tf
from helper.tf_layers_nets import fc_network

def inference(x, state_dim, layer_sizes=[512, 512, 512, 512], name="FFN", act='relu'):
    logit = fc_network(x, [state_dim]+layer_sizes+[1], name, act)
    return tf.nn.sigmoid(logit), logit
