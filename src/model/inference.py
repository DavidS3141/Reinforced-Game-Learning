#!/user/bin/env python

'''inference.py: Implement the network in tensorflow to make predictions.'''

################################################################################

import tensorflow as tf
from ..helper.tf_layers_nets import fc_network

def inference(x, layer_sizes=[27, 50, 50, 1], name="FFN", act='relu'):
    return fc_network(x, layer_sizes, name, act)
