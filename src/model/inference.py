#!/user/bin/env python

'''inference.py: Implement the network in tensorflow to make predictions.'''

################################################################################

import tensorflow as tf
from helper.tf_layers_nets import fc_network

def inference(x, layer_sizes=[27, 512, 512, 512, 512, 1], name="FFN", act='relu'):
    logit = fc_network(x, layer_sizes, name, act)
    return tf.nn.sigmoid(logit), logit
