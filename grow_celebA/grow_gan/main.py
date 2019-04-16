import os
import scipy.misc
import numpy as np
from glob import glob

from growGAN import growGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

z_dims = '8.8.16.16.32.32.64.64.128.128.256'
epochs = '1.1.1.1.1.1.1.1.1.1.1'
g_layers = '2.4.4.6.6.8.8.10.10.12.12'
d_layers = '3.5.5.7.7.9.9.11.11.13.13'
output_dims = '4.8.8.16.16.32.32.64.64.128.128' 
useAlpha = 'n.y.n.y.n.y.n.y.n.y.n'
useBeta = 'n.n.y.n.y.n.y.n.y.n.y'
# useBeta = 'n.n.n.n.n.n.n.n.n.n.n'
feature_map_shrink = 'n' # ['n', 'f'] generator
feature_map_growth = 'n' # ['n', 'f'] discriminator
spatial_map_shrink = 'n' # ['n', 'f'] discriminator
spatial_map_growth = 'n' # ['n', 'f'] generator
stage = 'f.i.f.i.f.i.f.i.f.i.f'
loss = 'ns' # ['RaLS', 'ns', 'wa']
z_distr = 'u' # ['u', 'g']
activation = 'lrelu'
weight_init = 'z' # ['z', 'u', 'g', 'x', 'he']
lr = 0.0002
beta1 = 0.0
beta2 = 0.99
epsilon = 0.00000001
batch_size = 1
sample_num = 64
gpu = 1
g_batchnorm = True
d_batchnorm = True
normalize_z = False
crop = True
trainflag = True
visualize = False


model_dir = z_dims +'_'+ epochs +'_'+ g_layers +'_'+ d_layers +'_'+ output_dims +'_'+feature_map_shrink+feature_map_growth+spatial_map_shrink+spatial_map_growth+'_'+ loss +'_'+z_distr +'_'+ activation +'_'+ weight_init +'_'+ str(batch_size) +'_'+str(g_batchnorm) +'_'+ str(d_batchnorm) +'_'+ str(normalize_z)

flags = tf.app.flags
flags.DEFINE_string("z_dims", z_dims,
    "List of the latent space dimension per network cycle")
flags.DEFINE_string("epochs", epochs, 
    "List of epochs to train each network cycle with")
flags.DEFINE_string("g_layers", g_layers, 
    "List of layers to train each generator network cycle with")
flags.DEFINE_string("d_layers", d_layers, 
    "List of layers to train each discriminator network cycle with")
flags.DEFINE_string("output_dims", output_dims,
    "List of output dimensions to train each generator network cycle with")
flags.DEFINE_string("useAlpha", useAlpha,
    "Use spatial smoothing or not")
flags.DEFINE_string("useBeta", useBeta,
    "Use feature channel smoothing or not")
flags.DEFINE_string("feature_map_shrink", feature_map_shrink,
    "How fast the nbr of feature maps should decrease in the generator")
flags.DEFINE_string("feature_map_growth", feature_map_growth,
    "How fast the nbr of feature maps should increase in the discriminator")
flags.DEFINE_string("spatial_map_shrink", spatial_map_shrink,
    "How fast the spatial size should decrease in the discriminator")
flags.DEFINE_string("spatial_map_growth", spatial_map_growth,
    "How fast the spatial size should increase in the generator")
flags.DEFINE_string("stage", stage,
    "What stage the gan is at")
flags.DEFINE_string("loss", loss,
    "Loss function")
flags.DEFINE_string("z_distr", z_distr,
    "The latent distribution")
flags.DEFINE_string("weight_init", weight_init,
    "Weight initialization method when growing")
flags.DEFINE_string("activation", activation,
    "Activation function")
flags.DEFINE_float("lr", lr,
    "Learning rate of for adam")
flags.DEFINE_float("beta1", beta1, "Momentum term 1 of adam")
flags.DEFINE_float("beta2", beta2, "Momentum term 2 of adam")
flags.DEFINE_float("epsilon", epsilon, "Epsilon term of adam")
flags.DEFINE_integer("batch_size", batch_size, "The size of batch images")
flags.DEFINE_integer("sample_num", sample_num, "The size of sample images")
flags.DEFINE_integer("gpu", gpu, "GPU to use")
flags.DEFINE_boolean("g_batchnorm", g_batchnorm,
                     "True using batch norm, False for not using it")
flags.DEFINE_boolean("d_batchnorm", d_batchnorm,
                     "True using batch norm, False for not using it")
flags.DEFINE_boolean(
    "normalize_z", normalize_z, "sample on a hypersphere or not")
flags.DEFINE_boolean(
    "crop", crop, "crop the images to appropriate size")
flags.DEFINE_boolean(
    "trainflag", trainflag, "True for training, False for testing")
flags.DEFINE_boolean("visualize", visualize,
                     "True for visualizing, test mode")
flags.DEFINE_string("model_dir", model_dir,
                     "Directory name to save the images/models/logs")

FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    gan = growGAN(
        z_dims = FLAGS.z_dims,
        epochs = FLAGS.epochs,
        g_layers = FLAGS.g_layers,
        d_layers = FLAGS.d_layers,
        output_dims = FLAGS.output_dims,
        useAlpha = FLAGS.useAlpha,
        useBeta = FLAGS.useBeta,
        feature_map_shrink = FLAGS.feature_map_shrink,
        feature_map_growth = FLAGS.feature_map_growth,
        spatial_map_shrink = FLAGS.spatial_map_shrink,
        spatial_map_growth = FLAGS.spatial_map_growth,
        stage = FLAGS.stage,
        loss = FLAGS.loss,
        z_distr = FLAGS.z_distr,
        activation = FLAGS.activation,
        weight_init = FLAGS.weight_init,
        lr = FLAGS.lr,
        beta1 = FLAGS.beta1,
        beta2 = FLAGS.beta2,
        epsilon = FLAGS.epsilon,
        batch_size = FLAGS.batch_size,
        sample_num = FLAGS.sample_num,
        gpu = FLAGS.gpu,
        g_batchnorm = FLAGS.g_batchnorm,
        d_batchnorm = FLAGS.d_batchnorm,
        normalize_z = FLAGS.normalize_z,
        crop = FLAGS.crop,
        trainflag = FLAGS.trainflag,
        visualize = FLAGS.visualize,
        model_dir = FLAGS.model_dir) 

    show_all_variables()


    if FLAGS.trainflag:
        gan.train()
    else:
       if not gan.load()[0]:
           raise Exception("[!] Train a model first, then run test mode")

    if FLAGS.visualize:
        visualize(sess, gan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
