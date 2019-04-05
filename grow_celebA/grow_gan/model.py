import tensorflow as tf
from ops import *
from utils import *

# feature_map_shrink can be normal, fast. Normal is that we decrease the feature maps by half every other layer.
# fast is that we decrease them as late as possible, doing it for every layer when we need to.
# spatial_map_growth can be normal, fast. Normal is that we double the spatial dimension every other layer.
# fast is that we double the spatial dimension every layer.

def G(z, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', output_dim = 128,
    feature_map_shrink = 'n', spatial_map_growth = 'n'):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        if feature_map_shrink == 'f':
            nbr_layers_shrink = int(z.get_shape()[-1])//8
            idx_shrink = layers - np.log2(nbr_layers_shrink)
        print('input shape z:', z.get_shape())
        for i in range(layers):
            if i == 0:
                # fully-connected layers (equivalent to 4x4 conv)
                h = conv4x4(z, int(z.get_shape()[-1])*4*4, batch_size, name = 'g_h'+str(i+1))
                print('g_h1:', h.get_shape())
            else:
                if spatial_map_growth == 'n' and i % 2 == 0 and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                    print('upsampling')
                elif spatial_map_growth == 'f' and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                if feature_map_shrink == 'n':
                    if i % 2 == 0 and int(h.get_shape()[-1]) > 8:
                        h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('g_h'+str(i+1)+':', h.get_shape())
                        print('conv')
                    else:
                        h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('g_h'+str(i+1)+':', h.get_shape())
                elif feature_map_shrink == 'f':
                    if i >= idx_shrink:
                        h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('g_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('g_h'+str(i+1)+':', h.get_shape())

                
            if bn:
                g_bn = batch_norm(name='g_bn'+str(i+1))
                h = g_bn(h)
            if activation == 'lrelu':
                h = lrelu(h)
            elif activation == 'relu':
                h = relu(h)

        out = conv2d(h, 3, 1, 1, 1, 1, name='g_h'+str(layers+1), stddev = np.sqrt(2/(8*output_dim*output_dim)))
        print('out generator shape: ', out.get_shape())
        out = tf.nn.tanh(out)
    return out

# feature_map_growth can be normal, fast. Normal is that we increase the feature maps by doubling every other layer.
# fast is that we decrease them as early as possible, doing it for every layer up to 256.
# spatial_map_shrink can be normal, fast. Normal is that we halve the spatial dimension every other layer.
# fast is that we halve the spatial dimension every layer.


def D(image, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', input_dim = 128,
    feature_map_growth = 'n', spatial_map_shrink = 'n'):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        for i in range(layers):
            if i == 0:
                # 1x1 conv
                h = conv2d(image, 8, 1, 1, 1, 1, name = 'd_h1')
                print('d_h1:', h.get_shape())
            elif i == layers-1:
                h = conv2dVALID(h, int(h.get_shape()[-1]), 4, 4, 1, 1, name = 'd_h'+str(layers+1))
                print('d_h'+str(i+1)+':', h.get_shape())
            else:
                if spatial_map_shrink == 'n' and (i+1) % 2 == 0 and i != 1 and int(h.get_shape()[1]) > 4:
                    h = downscale2d(h, factor=2)
                elif spatial_map_shrink == 'fast' and int(h.get_shape()[1]) > 4:
                    h = downscale2d(h, factor=2)
                if feature_map_growth == 'n':
                    if i % 2 == 0 and int(h.get_shape()[-1]) < 256:
                        h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('d_h'+str(i+1)+':', h.get_shape())
                elif feature_map_growth == 'f':
                    if int(h.get_shape()[-1]) < 256:
                        h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('d_h'+str(i+1)+':', h.get_shape())

            if bn:
                d_bn = batch_norm(name='d_bn'+str(i+1))
                h = d_bn(h)
            if activation == 'lrelu':
                h = lrelu(h)
            elif activation == 'relu':
                h = relu(h)

        out = tf.layers.dense(h, 1, activation=None, name = 'd_h'+str(layers+1),
            kernel_initializer=tf.random_normal_initializer(0,stddev=np.sqrt(2/int(h.get_shape()[-1]))),bias_initializer=tf.zeros_initializer())
        print('d_h'+str(layers+1)+':', out.get_shape())
    return tf.nn.sigmoid(out), out