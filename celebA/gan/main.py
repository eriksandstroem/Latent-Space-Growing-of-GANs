import os
import scipy.misc
import numpy as np
from glob import glob

from GAN import GAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [30]")
flags.DEFINE_integer("gpu", 1, "GPU to use [1]")
flags.DEFINE_float("learning_rate", 0.0002,
                   "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_num", 64, "The size of sample images [64]")
flags.DEFINE_integer(
    "input_height",
    128,
    "The size of image to use (will be center cropped). [128]")
flags.DEFINE_integer(
    "input_width",
    None,
    "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 128,
                     "The size of the output images to produce [128]")
flags.DEFINE_integer("z_dim", 64,
                     "The size of the latent space [256]")
flags.DEFINE_integer(
    "output_width",
    None,
    "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA",
                    "The name of dataset [celebA]")
flags.DEFINE_string("loss", "RaLS",
                    "The name of loss function [RaLS, wa, ns]")
flags.DEFINE_string("zdistribution", "u",
                    "The name of latent distribution [g, u]")
flags.DEFINE_string("input_fname_pattern", "*.jpg",
                    "Glob pattern of filename of input images [*]")
flags.DEFINE_string("sample_dir", "train_samples",
                    "Directory name to save the image samples [samples]")
flags.DEFINE_string("architecture", "standard",
                    "Architecture used during training [standard]")
flags.DEFINE_boolean(
    "train", True, "True for training, False for testing [True]")
flags.DEFINE_boolean(
    "crop", True, "True for training, False for testing [True]")
flags.DEFINE_boolean("visualize", True,
                     "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.visible_device_list=str(FLAGS.gpu)

    with tf.Session(config=run_config) as sess:
        gan = GAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.sample_num,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            sample_dir=FLAGS.sample_dir,
            architecture = FLAGS.architecture,
            zdistribution = FLAGS.zdistribution,
            loss = FLAGS.loss,
            z_dim = FLAGS.z_dim) 

        show_all_variables()


        if FLAGS.train:
            gan.train(FLAGS)
        else:
           if not gan.load()[0]:
               raise Exception("[!] Train a model first, then run test mode")

        if FLAGS.visualize:
            visualize(sess, gan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
