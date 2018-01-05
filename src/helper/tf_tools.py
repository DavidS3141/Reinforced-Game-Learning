'''
helper/tf_tools.py: Define some helper functions for the tensorflow framework.
'''
###############################################################################

from input_spec import transform_data, detransform_data

import tensorflow as tf

optimizers = {'adam': tf.train.AdamOptimizer,
              'rmsprop': tf.train.RMSPropOptimizer}


def merge_tensors(A, B, axis=0, name='merge_tensors'):
    with tf.name_scope(name):
        merge_result = tf.concat([A, B], axis=axis)
        splitting_numbers = [tf.shape(A)[axis], tf.shape(B)[axis]]
        labelA = tf.zeros([splitting_numbers[0]], dtype=tf.float32)
        labelB = tf.ones([splitting_numbers[1]], dtype=tf.float32)
        label = tf.concat([labelA, labelB], axis=0)
        return merge_result, label, splitting_numbers


def split_tensor(tensor, splitting_numbers, axis=0, name='spit_tensor'):
    with tf.name_scope(name):
        A, B = tf.split(tensor, splitting_numbers, axis)
        return A, B


def apply_noise(in_tensor, noise):
    assert noise.get_shape()[0] == 1
    assert in_tensor.get_shape()[1] == noise.get_shape()[1]
    with tf.name_scope('apply_noise'):
        return in_tensor + tf.random_normal(tf.shape(in_tensor)) * noise


def apply_feat_transform(in_tensor, detransform=False):
    if not detransform:
        with tf.name_scope('transform_data'):
            return transform_data(in_tensor)
    else:
        with tf.name_scope('detransform_data'):
            return detransform_data(in_tensor)
