import os
import scipy.misc
import numpy as np
from glob import glob

from growGAN import growGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

z_dims = '256.256.256.256.256.256'
epochs = '4.8.8.8.8.8'
g_layers = '2.4.6.8.10.12'
d_layers = '3.5.7.9.11.13'
output_dims = '4.8.16.32.64.128' 
useAlpha = 'n.y.y.y.y.y'
useBeta = 'n.n.n.n.n.n'
stage = 'i.i.i.i.i.i'

# ----------------------------------
# Exact crippled pro settings
# z_dims = '512.512.512.512.512.512'
# epochs = '3.6.6.6.6.6'
# g_layers = '2.4.6.8.10.12'
# d_layers = '3.5.7.9.11.13'
# output_dims = '4.8.16.32.64.128' 
# useAlpha = 'n.y.y.y.y.y'
# useBeta = 'n.n.n.n.n.n'
# stage = 'i.i.i.i.i.i'
# feature_map_shrink = 'cpro' # ['n', 'f'] generator
# feature_map_growth = 'cpro' # ['n', 'f'] discriminator
# spatial_map_shrink = 'n' # ['n', 'f'] discriminator
# spatial_map_growth = 'n' # ['n', 'f'] generator
# -----------------------------------

# z_dims = '256'
# epochs = '44'
# g_layers = '4'
# d_layers = '5'
# output_dims = '8' 
# useAlpha = 'n'
# useBeta = 'n'
# stage = 'i'

feature_map_shrink = 'n' # ['n', 'f'] generator
feature_map_growth = 'n' # ['n', 'f'] discriminator
spatial_map_shrink = 'n' # ['n', 'f'] discriminator
spatial_map_growth = 'n' # ['n', 'f'] generator

loss = 'wa' # ['RaLS', 'ns', 'wa']
z_distr = 'g' # ['u', 'g']
activation = 'lrelu'
weight_init = 'z' # ['z', 'u', 'g', 'x', 'he']
lr = 0.0001
beta1 = 0.0
beta2 = 0.99
epsilon = 0.00000001
batch_size = 16  # REMEMBER TO CHECK THIS WHEN RUNNING!!!
sample_num = 64
gpu = 0
g_batchnorm = True
d_batchnorm = True
normalize_z = True
crop = True
trainflag = True
visualize = False
minibatch_std = True
use_wscale = False
use_pixnorm = False
D_loss_extra = False
G_run_avg = True
# THEN ONLY ADAPTIVE BATCH SIZE LEFT



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
flags.DEFINE_boolean("minibatch_std", minibatch_std,
                     "True using minibatch_std")
flags.DEFINE_boolean("use_wscale", use_wscale,
                     "True using use_wscale")
flags.DEFINE_boolean("use_pixnorm", use_pixnorm,
                     "True using use_pixnorm")
flags.DEFINE_boolean("D_loss_extra", D_loss_extra,
                     "True using D_loss_extra")
flags.DEFINE_boolean("G_run_avg", G_run_avg,
                     "True using G_run_avg")
# flags.DEFINE_string("model_dir", model_dir,
#                      "Directory name to save the images/models/logs")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    model_dir = str(FLAGS.lr)+'_'+FLAGS.z_dims +'_'+ FLAGS.epochs +'_'+ FLAGS.g_layers +'_'+ FLAGS.d_layers +'_'+ FLAGS.output_dims +'_'+FLAGS.feature_map_shrink+FLAGS.feature_map_growth+FLAGS.spatial_map_shrink+FLAGS.spatial_map_growth+'_'+ FLAGS.loss +'_'+FLAGS.z_distr +'_'+ FLAGS.activation +'_'+ FLAGS.weight_init +'_'+ str(FLAGS.batch_size) +'_'+str(FLAGS.g_batchnorm) +'_'+ str(FLAGS.d_batchnorm) +'_'+ str(FLAGS.normalize_z)+'_'+ str(FLAGS.minibatch_std) +'_'+str(FLAGS.use_wscale) +'_'+ str(FLAGS.use_pixnorm) +'_'+ str(FLAGS.D_loss_extra)


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
        model_dir = model_dir,
        minibatch_std = FLAGS.minibatch_std,
        use_wscale = FLAGS.use_wscale,
        use_pixnorm = FLAGS.use_pixnorm,
        D_loss_extra = FLAGS.D_loss_extra,
        G_run_avg = FLAGS.G_run_avg) 

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
