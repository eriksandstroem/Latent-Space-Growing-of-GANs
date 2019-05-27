import os
import scipy.misc
import numpy as np
from glob import glob

from GAN import GAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

z_dims = 8
epochs = 20
g_layers = 12
d_layers = 13
output_dims = 128

feature_map_shrink = 'n' # ['n', 'f', 'pro', 'cpro'] generator
feature_map_growth = 'n' # ['n', 'f', 'pro', 'cpro'] discriminator
spatial_map_shrink = 'n' # ['n', 'f'] discriminator
spatial_map_growth = 'n' # ['n', 'f'] generator

loss = 'wa' # ['RaLS', 'ns', 'wa']
z_distr = 'g' # ['u', 'g']
activation = 'lrelu'
lr = 0.0001 #<-- old lr
beta1 = 0.0
beta2 = 0.99
epsilon = 0.00000001
batch_size = 16  # REMEMBER TO CHECK THIS WHEN RUNNING!!!
sample_num = 64
gpu = 1
normalize_z = True
crop = True
minibatch_std = True
use_wscale = False
use_pixnorm = False
D_loss_extra = False



flags = tf.app.flags
flags.DEFINE_integer("z_dims", z_dims,
    "List of the latent space dimension per network cycle")
flags.DEFINE_integer("epochs", epochs, 
    "List of epochs to train each network cycle with")
flags.DEFINE_integer("g_layers", g_layers, 
    "List of layers to train each generator network cycle with")
flags.DEFINE_integer("d_layers", d_layers, 
    "List of layers to train each discriminator network cycle with")
flags.DEFINE_integer("output_dims", output_dims,
    "List of output dimensions to train each generator network cycle with")
flags.DEFINE_string("feature_map_shrink", feature_map_shrink,
    "How fast the nbr of feature maps should decrease in the generator")
flags.DEFINE_string("feature_map_growth", feature_map_growth,
    "How fast the nbr of feature maps should increase in the discriminator")
flags.DEFINE_string("spatial_map_shrink", spatial_map_shrink,
    "How fast the spatial size should decrease in the discriminator")
flags.DEFINE_string("spatial_map_growth", spatial_map_growth,
    "How fast the spatial size should increase in the generator")
flags.DEFINE_string("loss", loss,
    "Loss function")
flags.DEFINE_string("z_distr", z_distr,
    "The latent distribution")
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
flags.DEFINE_boolean(
    "normalize_z", normalize_z, "sample on a hypersphere or not")
flags.DEFINE_boolean(
    "crop", crop, "crop the images to appropriate size")
flags.DEFINE_boolean("minibatch_std", minibatch_std,
                     "True using minibatch_std")
flags.DEFINE_boolean("use_wscale", use_wscale,
                     "True using use_wscale")
flags.DEFINE_boolean("use_pixnorm", use_pixnorm,
                     "True using use_pixnorm")
flags.DEFINE_boolean("D_loss_extra", D_loss_extra,
                     "True using D_loss_extra")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    # model_dir = str(FLAGS.lr)+'_'+str(FLAGS.z_dims) +'_'+ str(FLAGS.epochs) +'_'+ str(FLAGS.g_layers) +'_'+ str(FLAGS.d_layers) +'_'+ str(FLAGS.output_dims) +'_'+FLAGS.feature_map_shrink+FLAGS.feature_map_growth+FLAGS.spatial_map_shrink+FLAGS.spatial_map_growth+'_'+ FLAGS.loss +'_'+FLAGS.z_distr +'_'+ FLAGS.activation +'_'+ str(FLAGS.batch_size) +'_'+ str(FLAGS.normalize_z)+'_'+ str(FLAGS.minibatch_std) +'_'+str(FLAGS.use_wscale) +'_'+ str(FLAGS.use_pixnorm) +'_'+ str(FLAGS.D_loss_extra)
    model_dir = str(FLAGS.lr)+'_'+str(FLAGS.z_dims) +'_44' +'_'+ str(FLAGS.g_layers) +'_'+ str(FLAGS.d_layers) +'_'+ str(FLAGS.output_dims) +'_'+FLAGS.feature_map_shrink+FLAGS.feature_map_growth+FLAGS.spatial_map_shrink+FLAGS.spatial_map_growth+'_'+ FLAGS.loss +'_'+FLAGS.z_distr +'_'+ FLAGS.activation +'_'+ str(FLAGS.batch_size) +'_'+ str(FLAGS.normalize_z)+'_'+ str(FLAGS.minibatch_std) +'_'+str(FLAGS.use_wscale) +'_'+ str(FLAGS.use_pixnorm) +'_'+ str(FLAGS.D_loss_extra)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.visible_device_list=str(FLAGS.gpu)

    with tf.Session(config=run_config) as sess: 
        gan = GAN(
            sess = sess,
            z_dim = FLAGS.z_dims,
            epochs = FLAGS.epochs,
            g_layers = FLAGS.g_layers,
            d_layers = FLAGS.d_layers,
            feature_map_shrink = FLAGS.feature_map_shrink,
            feature_map_growth = FLAGS.feature_map_growth,
            spatial_map_shrink = FLAGS.spatial_map_shrink,
            spatial_map_growth = FLAGS.spatial_map_growth,
            loss = FLAGS.loss,
            z_distr = FLAGS.z_distr,
            activation = FLAGS.activation,
            lr = FLAGS.lr,
            beta1 = FLAGS.beta1,
            beta2 = FLAGS.beta2,
            epsilon = FLAGS.epsilon,
            batch_size = FLAGS.batch_size,
            sample_num = FLAGS.sample_num,
            input_size = 128,
            output_size = FLAGS.output_dims,
            normalize_z = FLAGS.normalize_z,
            crop = FLAGS.crop,
            model_dir = model_dir,
            minibatch_std = FLAGS.minibatch_std,
            use_wscale = FLAGS.use_wscale,
            use_pixnorm = FLAGS.use_pixnorm,
            D_loss_extra = FLAGS.D_loss_extra)

        gan.train()
        print('done training')


if __name__ == '__main__':
    tf.app.run()
