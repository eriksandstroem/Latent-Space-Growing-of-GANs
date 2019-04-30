import tensorflow as tf
from ops import *
from utils import *

# feature_map_shrink can be normal, fast. Normal is that we decrease the feature maps by half every other layer.
# fast is that we decrease them as late as possible, doing it for every layer when we need to.
# spatial_map_growth can be normal, fast. Normal is that we double the spatial dimension every other layer.
# fast is that we double the spatial dimension every layer.

def G(z, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', output_dim = 128,
    feature_map_shrink = 'n', spatial_map_growth = 'n', use_wscale = True, use_pixnorm = True):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        if feature_map_shrink == 'f':
            nbr_layers_shrink = int(z.get_shape()[-1])//8
            idx_shrink = layers - np.log2(nbr_layers_shrink)
            print('idx_shrink: ', idx_shrink)
        print('input shape z:', z.get_shape())
        for i in range(layers):
            if i == 0:
                # fully-connected layers (equivalent to 4x4 conv)
                print('batch or sample size: ', batch_size)
                h = act(conv4x4(z, int(z.get_shape()[-1])*4*4, batch_size, name = 'g_h'+str(i+1), use_wscale = use_wscale), activation)
                print('g_h1:', h.get_shape())
            else:
                if spatial_map_growth == 'n' and i % 2 == 0 and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)

                elif spatial_map_growth == 'f' and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                if feature_map_shrink == 'n':
                    if i % 2 == 0 and int(h.get_shape()[-1]) > 8:
                        h = act(conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        if use_pixnorm:
                            h = pixel_norm(h)
                        print('g_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = act(conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        if use_pixnorm:
                            h = pixel_norm(h)
                        print('g_h'+str(i+1)+':', h.get_shape())
                elif feature_map_shrink == 'f':
                    if i >= idx_shrink:
                        h = act(conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        if use_pixnorm:
                            h = pixel_norm(h)
                        print('g_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = act(conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        if use_pixnorm:
                            h = pixel_norm(h)                      
                        print('g_h'+str(i+1)+':', h.get_shape())

        out = conv2d(h, 3, 1, 1, 1, 1, name='g_out'+str(layers), use_wscale = use_wscale)

        print('out generator shape: ', out.get_shape())
        
        out = tf.nn.tanh(out)
    return out

# feature_map_growth can be normal, fast. Normal is that we increase the feature maps by doubling every other layer.
# fast is that we decrease them as early as possible, doing it for every layer up to 256.
# spatial_map_shrink can be normal, fast. Normal is that we halve the spatial dimension every other layer.
# fast is that we halve the spatial dimension every layer.


def D(image, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', input_dim = 128,
    feature_map_growth = 'n', spatial_map_shrink = 'n', z_dim = 8, minibatch_std = True, use_wscale = True):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        idx = layers
        downsampleList = []
        idx = idx - 2
        while idx > 1:
            downsampleList.append(idx)
            idx = idx - 2

        featureUpsampleList = np.array(downsampleList) - 1

        print('Indices when to downsample: ', downsampleList)

        featureDim = 256/np.power(2,(layers-3)//2)


        for i in range(layers):
            if i == 0:
                # 1x1 conv
                h = act(conv2d(image, featureDim, 1, 1, 1, 1, name = 'd_h1', use_wscale = use_wscale), activation)

                print('d_h1:', h.get_shape())
            elif i == layers-1:
                h = act(conv2d(h, int(h.get_shape()[-1]), 4, 4, 1, 1, name = 'd_h'+str(layers), padding = 'VALID', use_wscale = use_wscale), activation)
                print('d_h'+str(i+1)+':', h.get_shape())
            else:
                if spatial_map_shrink == 'n' and i in downsampleList and int(h.get_shape()[1]) > 4: #and i != 1 and (i+1) % 2 == 0
                    h = downscale2d(h, factor=2)
                elif spatial_map_shrink == 'f' and int(h.get_shape()[1]) > 4:
                    h = downscale2d(h, factor=2)
                if feature_map_growth == 'n':
                    if i in featureUpsampleList and int(h.get_shape()[-1]) < z_dim: # i % 2 == 0
                        h = act(conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        if i == layers -2 and minibatch_std:
                            h = minibatch_stddev_layer(h)
                            print('shape after minibatch_std: ', h.get_shape())
                        h = act(conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        print('d_h'+str(i+1)+':', h.get_shape())
                elif feature_map_growth == 'f':
                    if int(h.get_shape()[-1]) < z_dim:
                        h = act(conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        if i == layers -2 and minibatch_std:
                            h = minibatch_stddev_layer(h)
                        h = act(conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), padding = 'SAME', use_wscale = use_wscale), activation)
                        # print('d_h'+str(i+1)+':', h.get_shape())

        out = dense(h, 1, name = 'd_out', use_wscale = use_wscale)

        print('d_out:', out.get_shape())
    return tf.nn.sigmoid(out), out


