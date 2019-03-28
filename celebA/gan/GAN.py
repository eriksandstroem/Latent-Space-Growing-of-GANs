from __future__ import division
import os
import time
import math
from glob import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np
from six.moves import xrange

from model import *
from ops import *
from utils import *


class GAN(object):
    def __init__(
            self,
            sess,
            input_height=128,
            input_width=128,
            crop=True,
            batch_size=64,
            sample_num=64,
            output_height=128,
            output_width=128,
            z_dim=256,
            c_dim=3,
            dataset_name='default',
            input_fname_pattern='*.jpg',
            sample_dir='samples',
            architecture = 'standard',
            zdistribution = 'u',
            loss = 'RaLS',
            learning_rate = 0.0002):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [256]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.loss = loss
        self.architecture = architecture
        self.zdistribution = zdistribution
        self.learning_rate = learning_rate

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern


        self.data = glob(os.path.join(
            "../../celebA_dataset", self.dataset_name, self.input_fname_pattern))
        self.data.sort()
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(self.data)

        self.build_model()

    def build_model(self):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = generator(self.z, self.batch_size)

        self.D_real, self.D_real_logits = discriminator(inputs, self.batch_size, reuse=False)
        self.D_fake, self.D_fake_logits = discriminator(self.G, self.batch_size, reuse=True)

        """loss function"""
        # d_loss
        self.d_loss_real = tf.reduce_mean(
            tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits) - 1))
        self.d_loss_fake = tf.reduce_mean(
            tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits) + 1))
        self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2

        # g_loss
        self.g_loss = (tf.reduce_mean(tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits))) / 2 +
                       tf.reduce_mean(tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits))) / 2)

        #sampler
        self.inputs_sample = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='real_images_sample')
        inputs_sample = self.inputs_sample
        self.sampler = sampler(self.z, sample_num = self.sample_num)

        self.D_real_sample, self.D_real_logits_sample = discriminator(inputs_sample, self.sample_num, reuse=True)
        self.D_fake_sample, self.D_fake_logits_sample = discriminator(self.sampler, self.sample_num, reuse=True)
         # d_loss
        self.d_loss_real_sample = tf.reduce_mean(
            tf.square(self.D_real_logits_sample - tf.reduce_mean(self.D_fake_logits_sample) - 1))
        self.d_loss_fake_sample = tf.reduce_mean(
            tf.square(self.D_fake_logits_sample - tf.reduce_mean(self.D_real_logits_sample) + 1))
        self.d_loss_sample = (self.d_loss_real_sample + self.d_loss_fake_sample) / 2

        # g_loss
        self.g_loss_sample = (tf.reduce_mean(tf.square(self.D_fake_logits_sample - tf.reduce_mean(self.D_real_logits_sample))) / 2 +
                       tf.reduce_mean(tf.square(self.D_real_logits_sample - tf.reduce_mean(self.D_fake_logits_sample))) / 2)

        """data visualization"""
        self.z_sum = histogram_summary("z", self.z)
        self.d_real_sum = histogram_summary("d_real", self.D_real)
        self.d_fake_sum = histogram_summary("d_fake", self.D_fake)
        self.G_sum = image_summary("G", self.G)
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.d_loss,
            var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.g_loss,
            var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except BaseException:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d_fake_sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.d_real_sum, self.d_loss_real_sum, self.d_loss_sum])


        if not os.path.exists('../logs/'+self.model_dir):
            os.makedirs('../logs/'+self.model_dir)
        self.writer = SummaryWriter('../logs/'+self.model_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = self.data[0:self.sample_num]
        sample = [
            get_image(
                sample_file,
                input_height=self.input_height,
                input_width=self.input_width,
                resize_height=self.output_height,
                resize_width=self.output_width,
                crop=self.crop) for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)

        counter = 0
        start_time = time.time()
        could_load, model_counter = self.load()
        if could_load:
            counter = model_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            #self.data = glob(os.path.join(
            #    "../../celebA_dataset", config.dataset, self.input_fname_pattern))
            #self.data.sort()

            np.random.shuffle(self.data)

            batch_idxs = len(self.data) // config.batch_size


            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx * config.batch_size:
                                        (idx + 1) * config.batch_size]
                batch = [
                    get_image(
                        batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1, 1,
                                            [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
                    self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                                               self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to
                # zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                                               self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(
                    {self.inputs: batch_images, self.z: batch_z})
                errD_real = self.d_loss_real.eval(
                    {self.inputs: batch_images, self.z: batch_z})
                errG = self.g_loss.eval(
                    {self.inputs: batch_images, self.z: batch_z})

                
                print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                    (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 50) == 0:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss_sample, self.g_loss_sample],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs_sample: sample_inputs,
                        },
                    )
                    if not os.path.exists('../{}/{}'.format(config.sample_dir, self.model_dir)):
                        os.makedirs('../{}/{}'.format(config.sample_dir, self.model_dir))

                    save_images(
                        samples, 
                            [int(np.sqrt(self.sample_num)),int(np.sqrt(self.sample_num))], '../{}/{}/train_{:02d}_{:04d}.png'.format(
                                config.sample_dir, self.model_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                          (d_loss, g_loss))
                if np.mod(counter, 50) == 0:
                    self.save(counter)
                counter += 1

    @property
    def model_dir(self):
        return "Arch{}_Zd{}_L{}_Bs{}_Lr{}_Zd{}_Iwh{}_Owh{}".format(
            self.architecture, self.z_dim, self.loss, self.batch_size, self.learning_rate, self.zdistribution,
            self.input_height, self.output_width)

    def save(self, step):
        model_name = "model"

        if not os.path.exists('../models/'+self.model_dir):
            os.makedirs('../models/'+self.model_dir)

        self.saver.save(self.sess, '../models/' + self.model_dir + '/' + model_name, global_step=step)

    def load(self):
        import re
        print(" [*] Reading models...")

        ckpt = tf.train.get_checkpoint_state('../models/'+self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join('../models',
                self.model_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find the model")
            return False, 0
