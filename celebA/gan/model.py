import tensorflow as tf
from ops import *
from utils import *

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')
d_bn5 = batch_norm(name='d_bn5')
d_bn6 = batch_norm(name='d_bn6')
d_bn7 = batch_norm(name='d_bn7')
d_bn8 = batch_norm(name='d_bn8')
d_bn9 = batch_norm(name='d_bn9')
d_bn10 = batch_norm(name='d_bn10')
d_bn11 = batch_norm(name='d_bn11')
d_bn12 = batch_norm(name='d_bn12')
d_bn13 = batch_norm(name='d_bn13')

g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')
g_bn8 = batch_norm(name='g_bn8')
g_bn9 = batch_norm(name='g_bn9')
g_bn10 = batch_norm(name='g_bn10')
g_bn11 = batch_norm(name='g_bn11')
g_bn12 = batch_norm(name='g_bn12')

def generatorPRO(z, batch_size=64, reuse = False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        #print('z: ', z.get_shape())
        # fully-connected layers (equivalent to 4x4 conv)
        h1 = conv4x4(z, 256*4*4, batch_size, name = 'g_h1')
        #print('gh1: ', h1.get_shape())
        h1 = lrelu(g_bn1(h1))


        # conv and upsampling layers
        h2 = conv2d(h1, 256, 3, 3, 1, 1, name='g_h2', stddev = np.sqrt(2/(256*4*4)))
        h2 = lrelu(g_bn2(h2))
        #print('gh21: ', h2.get_shape())
        h2 = upscale2d(h2, factor=2)
        #print('gh22: ', h2.get_shape())

        # conv and upsampling layers
        h3 = conv2d(h2, 128, 3, 3, 1, 1, name='g_h3', stddev = np.sqrt(2/(256*8*8)))
        h3 = lrelu(g_bn3(h3))
        h4 = conv2d(h3, 128, 3, 3, 1, 1, name='g_h4', stddev = np.sqrt(2/(128*8*8)))
        h4 = lrelu(g_bn4(h4))
        #print('gh41: ', h4.get_shape())
        h4 = upscale2d(h4, factor=2)
        #print('gh41: ', h4.get_shape())

        # conv and upsampling layers
        h5 = conv2d(h4, 64, 3, 3, 1, 1, name='g_h5', stddev = np.sqrt(2/(128*16*16)))
        h5 = lrelu(g_bn5(h5))
        h6 = conv2d(h5, 64, 3, 3, 1, 1, name='g_h6', stddev = np.sqrt(2/(64*16*16)))
        h6 = lrelu(g_bn6(h6))
        h6 = upscale2d(h6, factor=2)

        # conv and upsampling layers
        h7 = conv2d(h6, 32, 3, 3, 1, 1, name='g_h7', stddev = np.sqrt(2/(64*32*32)))
        h7 = lrelu(g_bn7(h7))
        h8 = conv2d(h7, 32, 3, 3, 1, 1, name='g_h8', stddev = np.sqrt(2/(32*32*32)))
        h8 = lrelu(g_bn8(h8))
        h8 = upscale2d(h8, factor=2)

        # conv and upsampling layers
        h9 = conv2d(h8, 16, 3, 3, 1, 1, name='g_h9', stddev = np.sqrt(2/(32*64*64)))
        h9 = lrelu(g_bn9(h9))
        h10 = conv2d(h9, 16, 3, 3, 1, 1, name='g_h10', stddev = np.sqrt(2/(16*64*64)))
        h10 = lrelu(g_bn10(h10))
        h10 = upscale2d(h10, factor=2)

        # conv and upsampling layers
        h11 = conv2d(h10, 8, 3, 3, 1, 1, name='g_h11', stddev = np.sqrt(2/(16*128*128)))
        h11 = lrelu(g_bn11(h11))
        h12 = conv2d(h11, 8, 3, 3, 1, 1, name='g_h12', stddev = np.sqrt(2/(8*128*128)))
        h12 = lrelu(g_bn12(h12))

        h13 = conv2d(h12, 3, 1, 1, 1, 1, name='g_h13', stddev = np.sqrt(2/(8*128*128)))

        print('out generator shape: ', h13.get_shape())
        h13 = tf.nn.tanh(h13)

        return h13


def generatorPROwoBn(z, batch_size=64, reuse = False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        print('z: ', z.get_shape())
        # fully-connected layers (equivalent to 4x4 conv)
        h1 = conv4x4(z, 256*4*4, batch_size, name = 'g_h1')
        print('gh1: ', h1.get_shape())
        h1 = lrelu(h1)

        # conv and upsampling layers
        h2 = conv2d(h1, 256, 3, 3, 1, 1, name='g_h2', stddev = np.sqrt(2/(256*4*4)))
        h2 = lrelu(h2)
        #print('gh21: ', h2.get_shape())
        h2 = upscale2d(h2, factor=2)
        #print('gh22: ', h2.get_shape())

        # conv and upsampling layers
        h3 = conv2d(h2, 128, 3, 3, 1, 1, name='g_h3', stddev = np.sqrt(2/(256*8*8)))
        h3 = lrelu(h3)
        h4 = conv2d(h3, 128, 3, 3, 1, 1, name='g_h4', stddev = np.sqrt(2/(128*8*8)))
        h4 = lrelu(h4)
        #print('gh41: ', h4.get_shape())
        h4 = upscale2d(h4, factor=2)
        #print('gh41: ', h4.get_shape())

        # conv and upsampling layers
        h5 = conv2d(h4, 64, 3, 3, 1, 1, name='g_h5', stddev = np.sqrt(2/(128*16*16)))
        h5 = lrelu(h5)
        h6 = conv2d(h5, 64, 3, 3, 1, 1, name='g_h6', stddev = np.sqrt(2/(64*16*16)))
        h6 = lrelu(h6)
        h6 = upscale2d(h6, factor=2)

        # conv and upsampling layers
        h7 = conv2d(h6, 32, 3, 3, 1, 1, name='g_h7', stddev = np.sqrt(2/(64*32*32)))
        h7 = lrelu(h7)
        h8 = conv2d(h7, 32, 3, 3, 1, 1, name='g_h8', stddev = np.sqrt(2/(32*32*32)))
        h8 = lrelu(h8)
        h8 = upscale2d(h8, factor=2)

        # conv and upsampling layers
        h9 = conv2d(h8, 16, 3, 3, 1, 1, name='g_h9', stddev = np.sqrt(2/(32*64*64)))
        h9 = lrelu(h9)
        h10 = conv2d(h9, 16, 3, 3, 1, 1, name='g_h10', stddev = np.sqrt(2/(16*64*64)))
        h10 = lrelu(h10)
        h10 = upscale2d(h10, factor=2)

        # conv and upsampling layers
        h11 = conv2d(h10, 8, 3, 3, 1, 1, name='g_h11', stddev = np.sqrt(2/(16*128*128)))
        h11 = lrelu(h11)
        h12 = conv2d(h11, 8, 3, 3, 1, 1, name='g_h12', stddev = np.sqrt(2/(8*128*128)))
        h12 = lrelu(h12)

        h13 = conv2d(h12, 3, 1, 1, 1, 1, name='g_h13', stddev = np.sqrt(2/(8*128*128)))

        print('out generator shape: ', h13.get_shape())
        h13 = tf.nn.tanh(h13)

        return h13

def generatorwoDeConv(z, batch_size=64, reuse = False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        # fully-connected layer 
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        print('shapeh1: ', h1.get_shape())

        h2 = upscale2d(h1, factor=2)
        print('shapeh2: ', h2.get_shape())
        h2 = conv2d(h2, 256, 5, 5, 1, 1, name = 'g_h2', stddev = np.sqrt(2/(256*16*16)))
        h2 = tf.nn.relu(g_bn2(h2))
        print('shapeh2: ', h2.get_shape())

        h3 = conv2d(h2, 256, 5, 5, 1, 1, name = 'g_h3', stddev = np.sqrt(2/(256*16*16)))
        h3 = tf.nn.relu(g_bn3(h3))
        print('shapeh3: ', h3.get_shape())

        h4 = upscale2d(h3, factor=2)
        h4 = conv2d(h4, 256, 5, 5, 1, 1, name = 'g_h4', stddev = np.sqrt(2/(256*32*32)))
        h4 = tf.nn.relu(g_bn4(h4))
        print('shapeh4: ', h4.get_shape())

        h5 = conv2d(h4, 256, 5, 5, 1, 1, name = 'g_h5', stddev = np.sqrt(2/(256*32*32)))
        h5 = tf.nn.relu(g_bn5(h5))
        print('shapeh5: ', h5.get_shape())

        h6 = upscale2d(h5, factor=2)
        h6 = conv2d(h6, 128, 5, 5, 1, 1, name = 'g_h6', stddev = np.sqrt(2/(256*64*64)))
        h6 = tf.nn.relu(g_bn6(h6))
        print('shapeh6: ', h6.get_shape())

        h7 = upscale2d(h6, factor=2)
        h7 = conv2d(h7, 64, 5, 5, 1, 1, name = 'g_h7', stddev = np.sqrt(2/(128*128*128)))
        h7 = tf.nn.relu(g_bn7(h7))
        print('shapeh7: ', h7.get_shape())

        h8 = conv2d(h7, 3, 5, 5, 1, 1, name = 'g_h8', stddev = np.sqrt(2/(64*128*128)))
        h8 = tf.nn.relu(g_bn8(h8))
        print('shapeh8: ', h8.get_shape())

        h8 = tf.nn.tanh(h8)

        return h8


def generator(z, batch_size=64, reuse = False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        # fully-connected layer 
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        print(h1.get_shape())
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # conv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      5, 5, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      5, 5, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      5, 5, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      5, 5, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      5, 5, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))

        h8 = deconv2d(h7, [batch_size, 128, 128, 3],
                      5, 5, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def discriminatorPRO(image, batch_size=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        print('input_shape: ', image.get_shape())
        # 1x1 conv
        h1 = conv2d(image, 8, 1, 1, 1, 1, name = 'd_h1')
        print('h1: ', h1.get_shape())
        h1 = lrelu(d_bn1(h1))

        # conv and downsampling layers
        h2 = conv2d(h1, 8, 3, 3, 1, 1, name = 'd_h2')
        h2 = lrelu(d_bn2(h2))
        print('h2: ', h2.get_shape())
        h3 = conv2d(h2, 16, 3, 3, 1, 1, name = 'd_h3')
        h3 = lrelu(d_bn3(h3))
        print('h31: ',h3.get_shape())
        h3 = downscale2d(h3, factor = 2)
        print('h32: ',h3.get_shape())

        # conv and downsampling layers
        h4 = conv2d(h3, 16, 3, 3, 1, 1, name = 'd_h4')
        h4 = lrelu(d_bn4(h4))
        h5 = conv2d(h4, 32, 3, 3, 1, 1, name = 'd_h5')
        h5 = lrelu(d_bn5(h5))
        print('h51: ',h5.get_shape())
        h5 = downscale2d(h5, factor = 2)
        print('h52: ',h5.get_shape())

        # conv and downsampling layers
        h6 = conv2d(h5, 32, 3, 3, 1, 1, name = 'd_h6')
        h6 = lrelu(d_bn6(h6))
        h7 = conv2d(h6, 64, 3, 3, 1, 1, name = 'd_h7')
        h7 = lrelu(d_bn7(h7))
        print('h71: ',h7.get_shape())
        h7 = downscale2d(h7, factor = 2)
        print('h72: ',h7.get_shape())

        # conv and downsampling layers
        h8 = conv2d(h7, 64, 3, 3, 1, 1, name = 'd_h8')
        h8 = lrelu(d_bn8(h8))
        h9 = conv2d(h8, 128, 3, 3, 1, 1, name = 'd_h9')
        h9 = lrelu(d_bn9(h9))
        print('h91: ',h9.get_shape())
        h9 = downscale2d(h9, factor = 2)
        print('h92: ',h9.get_shape())

        # conv and downsampling layers
        h10 = conv2d(h9, 128, 3, 3, 1, 1, name = 'd_h10')
        h10 = lrelu(d_bn10(h10))
        h11 = conv2d(h10, 256, 3, 3, 1, 1, name = 'd_h11')
        h11 = lrelu(d_bn11(h11))
        h11 = downscale2d(h11, factor = 2)
        print('h11: ',h11.get_shape())


        # last conv and fully connected layers
        h12 = conv2d(h11, 256, 3, 3, 1, 1, name = 'd_h12')
        h12 = lrelu(d_bn12(h12))
        print('h12: ',h12.get_shape())
        h13 = conv2dVALID(h12, 256, 4, 4, 1, 1, name = 'd_h13')
        print('h13: ', h13.get_shape())
        h13 = lrelu(d_bn13(h13))

        out = tf.layers.dense(h13, 1, activation=None, name = 'd_final',
            kernel_initializer=tf.initializers.random_normal(0,stddev=np.sqrt(2/256)),bias_initializer=tf.zeros_initializer())

        return tf.nn.sigmoid(out), out


def discriminatorPROwoBn(image, batch_size=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        print('input_shape: ', image.get_shape())
        # 1x1 conv
        h1 = conv2d(image, 8, 1, 1, 1, 1, name = 'd_h1')
        print('h1: ', h1.get_shape())
        h1 = lrelu(h1)

        # conv and downsampling layers
        h2 = conv2d(h1, 8, 3, 3, 1, 1, name = 'd_h2')
        h2 = lrelu(h2)
        print('h2: ', h2.get_shape())
        h3 = conv2d(h2, 16, 3, 3, 1, 1, name = 'd_h3')
        h3 = lrelu(h3)
        print('h31: ',h3.get_shape())
        h3 = downscale2d(h3, factor = 2)
        print('h32: ',h3.get_shape())

        # conv and downsampling layers
        h4 = conv2d(h3, 16, 3, 3, 1, 1, name = 'd_h4')
        h4 = lrelu(h4)
        h5 = conv2d(h4, 32, 3, 3, 1, 1, name = 'd_h5')
        h5 = lrelu(h5)
        print('h51: ',h5.get_shape())
        h5 = downscale2d(h5, factor = 2)
        print('h52: ',h5.get_shape())

        # conv and downsampling layers
        h6 = conv2d(h5, 32, 3, 3, 1, 1, name = 'd_h6')
        h6 = lrelu(h6)
        h7 = conv2d(h6, 64, 3, 3, 1, 1, name = 'd_h7')
        h7 = lrelu(h7)
        print('h71: ',h7.get_shape())
        h7 = downscale2d(h7, factor = 2)
        print('h72: ',h7.get_shape())

        # conv and downsampling layers
        h8 = conv2d(h7, 64, 3, 3, 1, 1, name = 'd_h8')
        h8 = lrelu(h8)
        h9 = conv2d(h8, 128, 3, 3, 1, 1, name = 'd_h9')
        h9 = lrelu(h9)
        print('h91: ',h9.get_shape())
        h9 = downscale2d(h9, factor = 2)
        print('h92: ',h9.get_shape())

        # conv and downsampling layers
        h10 = conv2d(h9, 128, 3, 3, 1, 1, name = 'd_h10')
        h10 = lrelu(h10)
        h11 = conv2d(h10, 256, 3, 3, 1, 1, name = 'd_h11')
        h11 = lrelu(h11)
        h11 = downscale2d(h11, factor = 2)
        print('h11: ',h11.get_shape())


        # last conv and fully connected layers
        h12 = conv2d(h11, 256, 3, 3, 1, 1, name = 'd_h12')
        h12 = lrelu(h12)
        print('h12: ',h12.get_shape())
        h13 = conv2dVALID(h12, 256, 4, 4, 1, 1, name = 'd_h13')
        print('h13: ', h13.get_shape())
        h13 = lrelu(h13)

        out = tf.layers.dense(h13, 1, activation=None, name = 'd_final',
            kernel_initializer=tf.initializers.random_normal(0,stddev=np.sqrt(2/256)),bias_initializer=tf.zeros_initializer())

        return tf.nn.sigmoid(out), out



def discriminator(image, batch_size=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        h1 = conv2d(image, 64, 5, 5, 2, 2, name='d_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, 5, 5, 2, 2, name='d_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, 5, 5, 2, 2, name='d_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, 5, 5, 2, 2, name='d_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'd_h5_lin')
        h5 = lrelu(h5)

        h6 = linear(h5, 1, 'd_h6_lin')

        return tf.nn.sigmoid(h6), h6


# def sampler(z, sample_num=64):
#     with tf.variable_scope("generator") as scope:
#         scope.reuse_variables()

#         # fully-connected layers
#         h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
#         h1 = tf.reshape(h1, [sample_num, 8, 8, 256])
#         h1 = tf.nn.relu(g_bn1(h1))

#         # deconv layers
#         h2 = deconv2d(h1, [sample_num, 16, 16, 256],
#                       5, 5, 2, 2, name='g_h2')
#         h2 = tf.nn.relu(g_bn2(h2))

#         h3 = deconv2d(h2, [sample_num, 16, 16, 256],
#                       5, 5, 1, 1, name='g_h3')
#         h3 = tf.nn.relu(g_bn3(h3))

#         h4 = deconv2d(h3, [sample_num, 32, 32, 256],
#                       5, 5, 2, 2, name='g_h4')
#         h4 = tf.nn.relu(g_bn4(h4))

#         h5 = deconv2d(h4, [sample_num, 32, 32, 256],
#                       5, 5, 1, 1, name='g_h5')
#         h5 = tf.nn.relu(g_bn5(h5))

#         h6 = deconv2d(h5, [sample_num, 64, 64, 128],
#                       5, 5, 2, 2, name='g_h6')
#         h6 = tf.nn.relu(g_bn6(h6))

#         h7 = deconv2d(h6, [sample_num, 128, 128, 64],
#                       5, 5, 2, 2, name='g_h7')
#         h7 = tf.nn.relu(g_bn7(h7))

#         h8 = deconv2d(h7, [sample_num, 128, 128, 3],
#                       5, 5, 1, 1, name='g_h8')
#         h8 = tf.nn.tanh(h8)

#         return h8


# feature_map_shrink can be normal, fast. Normal is that we decrease the feature maps by half every other layer.
# fast is that we decrease them as late as possible, doing it for every layer when we need to.
# spatial_map_growth can be normal, fast. Normal is that we double the spatial dimension every other layer.
# fast is that we double the spatial dimension every layer.

def G(z, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', output_dim = 128,
    feature_map_shrink = 'normal', spatial_map_growth = 'normal', stage = 'final', alpha = 1):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        if feature_map_shrink == 'fast':
            nbr_layers_shrink = int(z.get_shape()[-1])//8
            idx_shrink = layers - np.log2(nbr_layers_shrink)
            print('idx_shrink: ', idx_shrink)
        print('input shape z:', z.get_shape())
        for i in range(layers):
            if i == 0:
                # fully-connected layers (equivalent to 4x4 conv)
                h = conv4x4(z, int(z.get_shape()[-1])*4*4, batch_size, name = 'g_h'+str(i+1))
                print('g_h1:', h.get_shape())
            else:
                if spatial_map_growth == 'normal' and i % 2 == 0 and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                    if i == layers - 2 and stage == 'intermediate':
                        res_connect = h
                elif spatial_map_growth == 'fast' and int(h.get_shape()[1]) < output_dim:
                    h = upscale2d(h, factor=2)
                if feature_map_shrink == 'normal':
                    if i % 2 == 0 and int(h.get_shape()[-1]) > 8:
                        h = conv2d(h, int(h.get_shape()[-1])//2, 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('g_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='g_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('g_h'+str(i+1)+':', h.get_shape())
                elif feature_map_shrink == 'fast':
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
        if stage == 'intermediate':
            res_connect = conv2d(res_connect, 3, 1, 1, 1, 1, name='g_h'+str(layers+1)+'res', stddev = np.sqrt(2/(8*output_dim*output_dim)))
            out = tf.add(res_connect*(1-alpha), out*alpha, name = 'g_smoothed')
            print('alpha: ', alpha)
            print('fused')
        print('out generator shape: ', out.get_shape())
        
        out = tf.nn.tanh(out)
    return out

# feature_map_growth can be normal, fast. Normal is that we increase the feature maps by doubling every other layer.
# fast is that we decrease them as early as possible, doing it for every layer up to 256.
# spatial_map_shrink can be normal, fast. Normal is that we halve the spatial dimension every other layer.
# fast is that we halve the spatial dimension every layer.


def D(image, batch_size=64, reuse = False, bn = True, layers = 12, activation = 'lrelu', input_dim = 128,
    feature_map_growth = 'normal', spatial_map_shrink = 'normal', stage = 'final', alpha = 1): # stage = ['final', 'intermediate'] (['f', 'i'])
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
                if stage == 'intermediate':
                    res_connect = h
                    res_connect = downscale2d(res_connect, factor = 2)
                print('d_h1:', h.get_shape())
            elif i == layers-1:
                h = conv2dVALID(h, int(h.get_shape()[-1]), 4, 4, 1, 1, name = 'd_h'+str(layers+1))
                print('d_h'+str(i+1)+':', h.get_shape())
            else:
                if spatial_map_shrink == 'normal' and i in downsampleList and int(h.get_shape()[1]) > 4: #and i != 1 and (i+1) % 2 == 0
                    h = downscale2d(h, factor=2)
                    if stage == 'intermediate' and i == 3:
                        h = tf.add(res_connect*(1-alpha), h*alpha, name = 'd_smoothed')
                        print('fused')
                elif spatial_map_shrink == 'fast' and int(h.get_shape()[1]) > 4:
                    h = downscale2d(h, factor=2)
                if feature_map_growth == 'normal':
                    if i in downsampleList and int(h.get_shape()[-1]) < 256 and stage == 'final': # i % 2 == 0
                        h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('d_h'+str(i+1)+':', h.get_shape())
                    elif i in downsampleList and int(h.get_shape()[-1]) < 256 and stage == 'intermediate': # i % 2 == 0
                        if int(h.get_shape()[1])*4 <= int(image.get_shape()[1]):
                            h = conv2d(h, int(h.get_shape()[-1])*2, 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        else:
                            h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                            np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
       
                        print('d_h'+str(i+1)+':', h.get_shape())
                    else:
                        h = conv2d(h, int(h.get_shape()[-1]), 3, 3, 1, 1, name='d_h'+str(i+1), stddev = 
                        np.sqrt(2/(int(h.get_shape()[-1])*int(h.get_shape()[1])*int(h.get_shape()[2]))))
                        print('d_h'+str(i+1)+':', h.get_shape())
                elif feature_map_growth == 'fast':
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