import tensorflow as tf
from ops import *
from utils import *


d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')

g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')


def generatorPRO(z, batch_size=64):
    with tf.variable_scope("generator") as scope:

        # fully-connected layers (equivalent to 4x4 conv)
        h1 = conv4x4(z, 256*4*4, name = 'g_h1')
        h1 = lrelu(g_bn1(h1))

        # conv and upsampling layers
        h2 = conv2d(h1, 256, 3, 3, 1, 1, name='g_h2', stddev = np.sqrt(2/(256*4*4)))
        h2 = lrelu(g_bn2(h2))
        h2 = upscale2d(h2, factor=2)

        # conv and upsampling layers
        h3 = conv2d(h2, 128, 3, 3, 1, 1, name='g_h3', stddev = np.sqrt(2/(256*8*8)))
        h3 = lrelu(g_bn3(h3))
        h4 = conv2d(h3, 128, 3, 3, 1, 1, name='g_h4', stddev = np.sqrt(2/(128*8*8)))
        h4 = lrelu(g_bn4(h4))
        h4 = upscale2d(h4, factor=2)

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

        h13 = conv2d(h12, 3, 1, 1, 1, 1, name='g_h13', stddev = np.sqrt(2/(16*128*128)))

        h13 = tf.nn.tanh(h13)

        return h13


def generator(z, batch_size=64):
    with tf.variable_scope("generator") as scope:

        # fully-connected layer 
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
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

        # 1x1 conv
        h1 = conv2d(image, 8, 1, 1, 1, 1, name = 'd_h1')
        h1 = lrelu(g_bn1(h1))

        # conv and downsampling layers
        h2 = conv2d(h1, 8, 3, 3, 1, 1, name = 'd_h2')
        h2 = lrelu(g_bn2(h2))
        h3 = conv2d(h2, 16, 3, 3, 1, 1, name = 'd_h3')
        h3 = lrelu(g_bn3(h3))
        h3 = downscale(h3, factor = 2)

        # conv and downsampling layers
        h4 = conv2d(h3, 16, 3, 3, 1, 1, name = 'd_h4')
        h4 = lrelu(g_bn4(h4))
        h5 = conv2d(h2, 32, 3, 3, 1, 1, name = 'd_h5')
        h5 = lrelu(g_bn5(h5))
        h5 = downscale(h5, factor = 2)

        # conv and downsampling layers
        h6 = conv2d(h5, 32, 3, 3, 1, 1, name = 'd_h6')
        h6 = lrelu(g_bn6(h6))
        h7 = conv2d(h6, 64, 3, 3, 1, 1, name = 'd_h7')
        h7 = lrelu(g_bn7(h7))
        h7 = downscale(h7, factor = 2)

        # conv and downsampling layers
        h8 = conv2d(h7, 32, 3, 3, 1, 1, name = 'd_h8')
        h8 = lrelu(g_bn8(h8))
        h9 = conv2d(h8, 64, 3, 3, 1, 1, name = 'd_h9')
        h9 = lrelu(g_bn9(h9))
        h9 = downscale(h9, factor = 2)

        # conv and downsampling layers
        h10 = conv2d(h9, 64, 3, 3, 1, 1, name = 'd_h10')
        h10 = lrelu(g_bn10(h10))
        h11 = conv2d(h10, 128, 3, 3, 1, 1, name = 'd_h11')
        h11 = lrelu(g_bn11(h11))
        h11 = downscale(h11, factor = 2)

        # conv and downsampling layers
        h12 = conv2d(h11, 128, 3, 3, 1, 1, name = 'd_h12')
        h12 = lrelu(g_bn12(h12))
        h13 = conv2d(h12, 256, 3, 3, 1, 1, name = 'd_h13')
        h13 = lrelu(g_bn13(h13))
        h13 = downscale(h13, factor = 2)

        # conv and downsampling layers
        h14 = conv2d(h13, 256, 3, 3, 1, 1, name = 'd_h14')
        h14 = lrelu(g_bn14(h14))

        h15 = cond2dVALID(h14, 256, 4, 4, 1, 1, name = 'd_h15')
        h15 = lrelu(g_bn15(h15))

        out = tf.layers.dense(h15, 1, activation=None, name = 'd_final',
            kernel_initializer=tf.truncated_normal_initializer(0,stddev=np.sqrt(2/256),
              bias_initializer=tf.zeros_initializer()))

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


def sampler(z, sample_num=64):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [sample_num, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [sample_num, 16, 16, 256],
                      5, 5, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [sample_num, 16, 16, 256],
                      5, 5, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [sample_num, 32, 32, 256],
                      5, 5, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [sample_num, 32, 32, 256],
                      5, 5, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [sample_num, 64, 64, 128],
                      5, 5, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        h7 = deconv2d(h6, [sample_num, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))

        h8 = deconv2d(h7, [sample_num, 128, 128, 3],
                      5, 5, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8
