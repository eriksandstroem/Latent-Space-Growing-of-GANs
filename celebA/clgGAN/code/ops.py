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

def minibatch_stddev_layer(x, useTau = 'n', tau = 0.5): # POTENTIALLY REMOVE GROUP SIZE THING
    with tf.variable_scope('MinibatchStddev'):
        if useTau == 'y':
            s = tf.shape(x)                                    
            y = tf.cast(x, tf.float32)                           
            y -= tf.reduce_mean(y, axis=0, keepdims=True) 
            y = tf.reduce_mean(tf.square(y), axis=0)
            y = tf.sqrt(y+ 1e-8)
            stdold = y[:,:,:s[3]//2] 
            stdnew = y[:,:,s[3]//2:]
            y = (1.-tau)*tf.reduce_mean(stdold)+tau*tf.reduce_mean(stdnew)                   
            y = tf.cast([[[[y]]]], x.dtype)                
            y = tf.tile(y, [s[0], s[1], s[2], 1])               
            return tf.concat([x, y], axis=3)      

        else:
            s = tf.shape(x)                                     
            y = tf.cast(x, tf.float32)                           
            y -= tf.reduce_mean(y, axis=0, keepdims=True) 
            y = tf.reduce_mean(tf.square(y), axis=0)
            y = tf.sqrt(y+ 1e-8)
            y = tf.reduce_mean(y)                   
            y = tf.cast([[[[y]]]], x.dtype)                
            y = tf.tile(y, [s[0], s[1], s[2], 1])               
            return tf.concat([x, y], axis=3)   


def dense(input_, output_dim, name,
            kernel_initializer, bias_initializer, useBeta = 'n', beta = 1):
    with tf.variable_scope(name):

        kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=kernel_initializer)
        biases = tf.get_variable(
        'bias', [output_dim], initializer=bias_initializer)
        if useBeta == 'y':
            not_new = kernel[0:input_.get_shape()[-1]//2,:]
            all_new = kernel[input_.get_shape()[-1]//2:,:]
            all_new = beta*all_new

            kernel = tf.concat((not_new, all_new), axis = 0, name = 'kernel')

        input_ = tf.reshape(input_, [-1,input_.get_shape()[-1]])
        dense = tf.matmul(input_, kernel)
        dense = tf.nn.bias_add(dense, biases)

        return dense

def conv4x4(input_, output_dim, batch_size, name, useBeta = 'n', beta = 1):
    with tf.variable_scope(name):
        fan_in = output_dim//(4*4)
        stddev = np.sqrt(2/fan_in).astype(np.float32)
        kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=tf.initializers.random_normal(0,stddev=stddev))
        biases = tf.get_variable(
        'biases', [output_dim//(4*4)], initializer=tf.constant_initializer(0.0))
        if useBeta == 'y':
            partially_new = kernel[0:input_.get_shape()[-1]//2,:]
            partially_new = tf.reshape(partially_new, [input_.get_shape()[-1]//2, 4, 4, output_dim//(4*4)])
            partially_new_old = partially_new[:,:,:, 0:output_dim//(4*4*2)]
            partially_new_new = partially_new[:,:,:, output_dim//(4*4*2):]
            partially_new_new = beta*partially_new_new
            partially_new = tf.concat((partially_new_old,partially_new_new), axis = 3)
            partially_new = tf.reshape(partially_new, [input_.get_shape()[-1]//2, output_dim])

            all_new = kernel[input_.get_shape()[-1]//2:,:]
            all_new = beta*all_new

            kernel = tf.concat((partially_new, all_new), axis = 0, name = 'kernel')

            biases_old = biases[0:output_dim//2]
            biases_new = biases[output_dim//2:]
            biases_new = beta*biases_new
            biases = tf.concat((biases_old,biases_new), axis = 0, name = 'biases')


        dense = tf.matmul(input_, kernel)
        # dense = tf.layers.dense(input_, output_dim, activation=None, name =None, use_bias=False,
        #     kernel_initializer=tf.initializers.random_normal(0,stddev=stddev))

        dense = tf.reshape(dense, [-1, 4, 4, output_dim//(4*4)]) #[batch_size, 256, 4, 4]
        dense = tf.nn.bias_add(dense, biases)
        #dense = apply_bias(dense)

        return dense

# def upscale2d(x, factor=2):
#     assert isinstance(factor, int) and factor >= 1
#     if factor == 1: return x
#     with tf.variable_scope('Upscale2D'):
#         s = x.shape
#         x = tf.reshape(x, [s[0], s[3], s[1], s[2]])
#         x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
#         x = tf.tile(x, [1, 1, 1, factor, 1, factor])
#         x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
#         x = tf.reshape(x, [s[0], s[1]*factor, s[2]*factor, s[3]])
#         return x

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

# def conv2d(input_, output_dim,
#            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#            name="conv2d"):
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#                             initializer=tf.initializers.random_normal(0,stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[
#                             1, d_h, d_w, 1], padding='SAME') #, data_format='NCHW')

#         biases = tf.get_variable(
#             'biases', [output_dim], initializer=tf.constant_initializer(0.0))
#         conv = tf.nn.bias_add(conv, biases)
#         #conv = apply_bias(conv, biases)

#         return conv

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2,
           name="conv2d", padding = 'SAME', useBeta = 'n', beta = 1, last = False, first =  False, minibstd = False):

    with tf.variable_scope(name):
        stddev = np.sqrt(2/(int(input_.get_shape()[-1])*int(input_.get_shape()[1])*int(input_.get_shape()[2])))
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.initializers.random_normal(0,stddev=stddev))
        biases = tf.get_variable(
                'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        if useBeta == 'y':
            # if inputchannels = x, outputchannels = x/2, then the OJ was in: x/2 out x/4. This means that x/4 filters of depth x are completely new. This means that x/4 filters are the rest 
            # and each filter has x/2 filter channels that come from the previously restored network. The last x/2 channels need to be multiplied by beta before the filter is used.
            # the filter shape is [3,3,in,out] = [3,3,x,x/2]. We can split this into the two tensors of x/4 filters each.
            if minibstd:
                partially_new = w[:,:,:,0:output_dim//2]
                if first == False: # not used for the first grown layer in the discriminator
                    partially_new_old = partially_new[:,:,0:input_.get_shape()[-1]//2,:]
                    partially_new_new = partially_new[:,:,input_.get_shape()[-1]//2:,:]
                    partially_new_new_block = beta*partially_new_new[:,:,:-1,:]
                    partially_new_new_std = partially_new_new[:,:,-1,:]
                    partially_new_new_std = tf.reshape(partially_new_new_std,[partially_new_new_std.shape[0],partially_new_new_std.shape[1],1,partially_new_new_std.shape[2]])
                    partially_new_new = tf.concat((partially_new_new_block, partially_new_new_std),axis = 2)
                    partially_new = tf.concat((partially_new_old,partially_new_new), axis = 2)

                all_new = w[:,:,:,output_dim//2:] # we have established that sometimes this isn't zero, even though we want it to be always zero. This can happen if we in the model (now only generator) calls useBeta when 
                # we shouldn't.
                if last == False: # not used for the last grown layer in the generator
                    all_new = tf.scalar_mul(beta,all_new)
                    biases_old = biases[0:output_dim//2]
                    biases_new = biases[output_dim//2:]
                    biases_new = beta*biases_new
                    biases = tf.concat((biases_old,biases_new), axis = 0, name = 'biases')

                w = tf.concat((partially_new, all_new), axis = 3, name = 'w')
            else:
                partially_new = w[:,:,:,0:output_dim//2]
                if first == False: # not used for the first grown layer in the discriminator
                    partially_new_old = partially_new[:,:,0:input_.get_shape()[-1]//2,:]
                    partially_new_new = partially_new[:,:,input_.get_shape()[-1]//2:,:]
                    partially_new_new = beta*partially_new_new
                    partially_new = tf.concat((partially_new_old,partially_new_new), axis = 2)

                all_new = w[:,:,:,output_dim//2:] # we have established that sometimes this isn't zero, even though we want it to be always zero. This can happen if we in the model (now only generator) calls useBeta when 
                # we shouldn't.
                if last == False: # not used for the last grown layer in the generator
                    all_new = tf.scalar_mul(beta,all_new)
                    biases_old = biases[0:output_dim//2]
                    biases_new = biases[output_dim//2:]
                    biases_new = beta*biases_new
                    biases = tf.concat((biases_old,biases_new), axis = 0, name = 'biases')

                w = tf.concat((partially_new, all_new), axis = 3, name = 'w')



        conv = tf.nn.conv2d(input_, w, strides=[
                            1, d_h, d_w, 1], padding=padding)


        conv = tf.nn.bias_add(conv, biases)

        return conv

def dense(input_, output_dim, name, useBeta = 'n', beta = 1):
    with tf.variable_scope(name):
        stddev = np.sqrt(2/int(input_.get_shape()[-1]))
        kernel = tf.get_variable('kernel', [input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(0,stddev=stddev))
        biases = tf.get_variable(
        'bias', [output_dim], initializer=tf.zeros_initializer())
        if useBeta == 'y':
            not_new = kernel[0:input_.get_shape()[-1]//2,:]
            all_new = kernel[input_.get_shape()[-1]//2:,:]
            all_new = beta*all_new

            kernel = tf.concat((not_new, all_new), axis = 0, name = 'kernel')

        input_ = tf.reshape(input_, [-1,input_.get_shape()[-1]])
        dense = tf.matmul(input_, kernel)
        dense = tf.nn.bias_add(dense, biases)

        return dense

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.constant_initializer(0.0))
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

# def conv2dVALID(input_, output_dim,
#            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#            name="conv2d"):
#     with tf.variable_scope(name):
#         w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#                             initializer=tf.initializers.random_normal(0,stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[
#                             1, d_h, d_w, 1], padding='VALID') #, data_format='NCHW')

#         biases = tf.get_variable(
#             'biases', [output_dim], initializer=tf.constant_initializer(0.0))
#         conv = tf.nn.bias_add(conv, biases)
#         #conv = apply_bias(conv, biases)

#         return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable(
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def relu(x, name="relu"):
    return tf.maximum(x, 0.0)

def linear(input_, output_size,
           scope=None, stddev=0.02, bias_start=0.0, with_w=False):
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


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# define cross entropy loss
def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x, labels=y)
    except BaseException:
        return tf.nn.sigmoid_cross_entropy_with_logits(
logits=x, targets=y)