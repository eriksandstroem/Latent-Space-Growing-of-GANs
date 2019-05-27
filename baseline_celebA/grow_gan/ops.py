import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except BaseException:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        bn_init = {
            'beta': tf.constant_initializer(0.0),
            'gamma': tf.constant_initializer(1.0),
            'moving_mean': tf.constant_initializer(0.0),
            'moving_variance': tf.constant_initializer(1.0),
}
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=0, #self.epsilon ska in här sen
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            param_initializers=bn_init
                                            )


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def act(x, activation):
    if activation == 'lrelu':
        return lrelu(x)              
    elif activation == 'relu':
        return relu(x)

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
      return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + epsilon)


def minibatch_stddev_layer(x, group_size=4): # POTENTIALLY REMOVE GROUP SIZE THING
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[3], s[1], s[2]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [NHW1]  Replicate over group and pixels.
        return tf.concat([x, y], axis=3)                        # [NHW(C+1)]  Append as new fmap.



def dense(input_, output_dim, name, use_wscale = True):
    with tf.variable_scope(name):
        stddev = np.sqrt(2/int(input_.get_shape()[-1]))
        if use_wscale:
            wscale = tf.constant(np.float32(stddev), name='wscale')
            kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(0,1))*wscale
        else:
            kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(0,stddev=stddev))
        biases = tf.get_variable(
        'bias', [output_dim], initializer=tf.zeros_initializer())

        input_ = tf.reshape(input_, [-1,input_.get_shape()[-1]])
        dense = tf.matmul(input_, kernel)
        dense = tf.nn.bias_add(dense, biases)

        return dense

def conv4x4(input_, output_dim, batch_size, name, use_wscale = True):
    with tf.variable_scope(name):
        fan_in = output_dim//(4*4)
        stddev = np.sqrt(2/fan_in)
        if use_wscale:
            wscale = tf.constant(np.float32(stddev), name='wscale')
            kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=tf.initializers.random_normal(0,1))*wscale
        else:
            kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=tf.initializers.random_normal(0,stddev=stddev))
        biases = tf.get_variable(
        'biases', [output_dim//(4*4)], initializer=tf.constant_initializer(0.0))


        dense = tf.matmul(input_, kernel)

        dense = tf.reshape(dense, [-1, 4, 4, output_dim//(4*4)])
        dense = tf.nn.bias_add(dense, biases)

        return dense

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, factor, factor, 1, 1])
        x = tf.reshape(x, [-1, s[1]*factor, s[2]*factor, s[3]]) # s[0] on first location in shape before
        return x

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, factor, factor, 1]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID') #, data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2,
           name="conv2d", padding = 'SAME', use_wscale = True):
    with tf.variable_scope(name):
        stddev = np.sqrt(2/(int(input_.get_shape()[-1])*int(input_.get_shape()[1])*int(input_.get_shape()[2])))
        if use_wscale:
            wscale = tf.constant(np.float32(stddev), name='wscale')
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.initializers.random_normal(0,1))*wscale
        else:
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.initializers.random_normal(0,stddev=stddev))
        biases = tf.get_variable(
                'biases', [output_dim], initializer=tf.constant_initializer(0.0))


        conv = tf.nn.conv2d(input_, w, strides=[
                            1, d_h, d_w, 1], padding=padding)


        conv = tf.nn.bias_add(conv, biases)

        return conv

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.constant_initializer(0.0))
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])



def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def relu(x, name="relu"):
    return tf.maximum(x, 0.0)

def linear(input_, output_size,
           scope=None, stddev=0.02, bias_start=0.0, with_w=False): # NOT USED
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def maxpool2d(x, k=2): # NOT USED
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2): # NOT USED
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# define cross entropy loss # NOT USED
def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x, labels=y)
    except BaseException:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x, targets=y)
