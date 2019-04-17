import tensorflow as tf
from ops import *
from utils import *

# feature_map_shrink can be normal, fast. Normal is that we decrease the feature maps by half every other layer.
# fast is that we decrease them as late as possible, doing it for every layer when we need to.
# spatial_map_growth can be normal, fast. Normal is that we double the spatial dimension every other layer.
# fast is that we double the spatial dimension every layer.

def G(z, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', output_dim = 128,
    feature_map_shrink = 'n', spatial_map_growth = 'n', alpha = 1, useAlpha = 'y', beta = 1, useBeta = 'y'):
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
                h = conv4x4(z, int(z.get_shape()[-1])*4*4, batch_size, name = 'g_h'+str(i+1), useBeta = useBeta, beta = beta)
                print('g_h1:', h.get_shape())
            else:
                if spatial_map_growth == 'n' and i % 2 == 0 and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                    if i == layers - 2 and useAlpha == 'y':
                        res_connect = h
                elif spatial_map_growth == 'f' and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                if feature_map_shrink == 'n':
                    if i % 2 == 0 and int(h.get_shape()[-1]) > 8:
                        if i <= layers - 2:
                            if i == layers - 2:
                                h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, last = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('g_h'+str(i+1)+':', h.get_shape())
                    else:
                        if i <= layers - 2:
                            if i == layers - 2:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, last = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('g_h'+str(i+1)+':', h.get_shape())
                elif feature_map_shrink == 'f':
                    if i >= idx_shrink:
                        if i <= layers - 2:
                            if i == layers - 2:
                                h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, last = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('g_h'+str(i+1)+':', h.get_shape())
                    else:
                        if i <= layers - 2:
                            if i == layers - 2:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, last = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('g_h'+str(i+1)+':', h.get_shape())

                
            # if bn:
            #     g_bn = batch_norm(name='g_bn'+str(i+1))
            #     h = g_bn(h)
            if activation == 'lrelu':
                h = lrelu(h)
            elif activation == 'relu':
                h = relu(h)

        if useAlpha == 'y':
            h = tf.add(tf.scalar_mul(1-alpha,res_connect), tf.scalar_mul(alpha,h), name = 'g_smoothed')
            print('fused')

        out = conv2d(h, 3, 1, 1, 1, 1, name='g_out', stddev = np.sqrt(2/(8*output_dim*output_dim)))

        print('out generator shape: ', out.get_shape())
        
        out = tf.nn.tanh(out)
    return out

# feature_map_growth can be normal, fast. Normal is that we increase the feature maps by doubling every other layer.
# fast is that we decrease them as early as possible, doing it for every layer up to 256.
# spatial_map_shrink can be normal, fast. Normal is that we halve the spatial dimension every other layer.
# fast is that we halve the spatial dimension every layer.


def D(image, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', input_dim = 128,
    feature_map_growth = 'n', spatial_map_shrink = 'n', stage = 'i', alpha = 1, useAlpha = 'y', beta = 1, useBeta = 'y', z_dim = 8):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        idx = layers
        downsampleList = []
        idx = idx - 2
        while idx > 1:
            downsampleList.append(idx)
            idx = idx - 2

        print('Indices when to downsample: ', downsampleList)

        for i in range(layers):
            if i == 0:
                # 1x1 conv
                h = conv2d(image, 8, 1, 1, 1, 1, name = 'd_h1')
                if useAlpha == 'y':
                    res_connect = h
                    if activation == 'lrelu':
                        res_connect = lrelu(res_connect)
                    elif activation == 'relu':
                        res_connect = relu(res_connect)
                    # BATCHNORM NEEDS TO BE HERE TOO!
                    res_connect = downscale2d(res_connect, factor = 2)

                print('d_h1:', h.get_shape())
            elif i == layers-1:
                h = conv2d(h, int(h.get_shape()[-1]), 4, 4, 1, 1, name = 'd_h'+str(layers), padding = 'VALID', useBeta = useBeta, beta = beta)
                print('d_h'+str(i+1)+':', h.get_shape())
            else:
                if spatial_map_shrink == 'n' and i in downsampleList and int(h.get_shape()[1]) > 4: #and i != 1 and (i+1) % 2 == 0
                    h = downscale2d(h, factor=2)
                    if useAlpha == 'y' and i == 3:
                        h = tf.add(tf.scalar_mul(1-alpha,res_connect), tf.scalar_mul(alpha, h), name = 'd_smoothed')
                        print('fused')
                elif spatial_map_shrink == 'f' and int(h.get_shape()[1]) > 4:
                    h = downscale2d(h, factor=2)
                if feature_map_growth == 'n':
                    if i in downsampleList and useAlpha == 'n' and int(h.get_shape()[-1]) < z_dim: # i % 2 == 0
                        if i >= 3:
                            if i == 3:
                                h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, first = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('d_h'+str(i+1)+':', h.get_shape())
                    elif i in downsampleList and int(h.get_shape()[-1]) < z_dim and stage == 'i': # i % 2 == 0
                        if int(h.get_shape()[1])*4 <= int(image.get_shape()[1]):
                            h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        else:
                            h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
       
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        if i >= 3:
                            if i == 3:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, first = True)
                            else:
                               h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta) 
                        else:
                            h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('d_h'+str(i+1)+':', h.get_shape())
                elif feature_map_growth == 'f':
                    if int(h.get_shape()[-1]) < z_dim:
                        if i >= 3:
                            if i == 3:
                                h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, first = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        if i >= 3:
                            if i == 3:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta, first = True)
                            else:
                                h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME', useBeta = useBeta, beta = beta)
                        else:
                            h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))), padding = 'SAME')
                        # print('d_h'+str(i+1)+':', h.get_shape())

            # if bn:
            #     d_bn = batch_norm(name='d_bn'+str(i+1))
            #     h = d_bn(h)
            if activation == 'lrelu':
                h = lrelu(h)
            elif activation == 'relu':
                h = relu(h)

        out = dense(h, 1, name = 'd_out',
            kernel_initializer=tf.random_normal_initializer(0,stddev=np.sqrt(2/int(h.get_shape()[-1]))),
            bias_initializer=tf.zeros_initializer(), useBeta = useBeta, beta = beta)

        print('d_out:', out.get_shape())
    return tf.nn.sigmoid(out), out

