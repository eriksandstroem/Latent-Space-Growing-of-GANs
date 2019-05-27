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

import matplotlib
matplotlib.use('agg')
from matplotlib import cm, pyplot as plt


class GAN(object):
	def __init__(
		self,
		sess,
		z_dim,
		epochs,
		g_layers,
		d_layers,
		feature_map_shrink,
		feature_map_growth,
		spatial_map_shrink,
		spatial_map_growth,
		loss,
		z_distr,
		activation,
		lr,
		beta1,
		beta2,
		epsilon,
		batch_size,
		sample_num,
		input_size,
		output_size,
		normalize_z,
		crop,
		model_dir,
		minibatch_std,
		use_wscale, 
		use_pixnorm,
		D_loss_extra):

		self.sess = sess
		self.z_dim = z_dim
		self.epochs = epochs
		self.g_layers = g_layers
		self.d_layers = d_layers
		self.feature_map_shrink = feature_map_shrink
		self.feature_map_growth = feature_map_growth
		self.spatial_map_shrink = spatial_map_shrink
		self.spatial_map_growth = spatial_map_growth
		self.loss = loss
		self.z_distr = z_distr
		self.activation = activation
		self.learning_rate = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.sample_num = sample_num
		self.input_size = 128
		self.output_size = output_size
		self.normalize_z = normalize_z
		self.crop = crop
		self.model_dir = model_dir
		self.model_dir_full = model_dir
		self.minibatch_std = minibatch_std
		self.use_wscale = use_wscale
		self.use_pixnorm = use_pixnorm
		self.D_loss_extra = D_loss_extra

		self.data = glob(os.path.join("../../celebA_dataset", 'celebA', '*.jpg'))
		self.data.sort()
		seed = 547
		np.random.seed(seed)
		np.random.shuffle(self.data)

		self.build_model()


	def build_model(self):
		if self.crop:
			image_dims = [self.output_size, self.output_size, 3] 
		else:
			image_dims = [self.input_size, self.input_size, 3]

		self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
		inputs = self.inputs

		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

		self.G = G(self.z, batch_size= self.batch_size, reuse = False, layers = self.g_layers, activation = self.activation, output_dim = self.output_size,
			feature_map_shrink = self.feature_map_shrink, spatial_map_growth = self.spatial_map_growth, use_wscale = self.use_wscale, use_pixnorm = self.use_pixnorm)
		self.D_real, self.D_real_logits = D(inputs, batch_size = self.batch_size, reuse = False, layers = self.d_layers, activation = self.activation,
		 input_dim = self.input_size, feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, z_dim = self.z_dim,
		  minibatch_std = self.minibatch_std, use_wscale = self.use_wscale)
		self.D_fake, self.D_fake_logits = D(self.G, batch_size = self.batch_size, reuse = True, layers = self.d_layers, activation = self.activation,
		 input_dim = self.input_size, feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, z_dim = self.z_dim,
		  minibatch_std = self.minibatch_std, use_wscale = self.use_wscale)


		"""loss function"""
		if self.loss == 'RaLS':
			# d_loss
			self.d_loss_real = tf.reduce_mean(
			    tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits) - 1))
			self.d_loss_fake = tf.reduce_mean(
			    tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits) + 1))
			self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2
			if self.D_loss_extra:
				self.d_loss = self.d_loss + 0.001*tf.reduce_mean(tf.square(self.D_real_logits))

			# g_loss
			self.g_loss = (tf.reduce_mean(tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits))) / 2 
				+ tf.reduce_mean(tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits))) / 2)
		elif self.loss == 'ns':
			self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits,labels=tf.ones_like(self.D_real_logits)))
			self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,labels=tf.zeros_like(self.D_fake_logits)))
			self.d_loss = self.d_loss_real + self.d_loss_fake
			if self.D_loss_extra:
				self.d_loss = self.d_loss + 0.001*tf.reduce_mean(tf.square(self.D_real_logits))
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,labels=tf.ones_like(self.D_fake_logits)))
		elif self.loss == 'wa':
			hyperparameter = 10
			gamma = tf.random_uniform(shape=[self.batch_size,1,1,1],minval=0., maxval=1.)
			#beta = tf.ones([batch_size,1,1,1],dtype=tf.float32)
			xhat = tf.add( tf.multiply(gamma,self.inputs), tf.multiply((1-gamma),self.G))

			_, D_xhat = D(xhat, batch_size = self.batch_size, reuse = True, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, z_dim = self.z_dim,
			 minibatch_std = self.minibatch_std, use_wscale = self.use_wscale)

			gradients = tf.gradients(D_xhat, xhat)[0] # is different between arch 1 and 2. Strange. The inputs are of different size, but just an upsampled version of the lower resolution version. Maybe that's why.
			# Since we take the gradient wrt xhat which is 4 times larger, when we sum the squares of gradients for each pixel we should get a larger gradient penalty for the larger image.
			#print('xhatshape', xhat.shape)_sample
			#print('idx: ', idx)
			#print('gradientdim', gradients) #(256,1,?,2) same as xhat
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
			#print('slpopedim:', slopes.shape) # (256,1)
			#gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)
			gradient_penalty = tf.reduce_mean((slopes-1.)**2)
			self.d_loss_fake = tf.reduce_mean(self.D_fake_logits)
			self.d_loss_real = -tf.reduce_mean(self.D_real_logits) + hyperparameter*gradient_penalty

			self.g_loss = -tf.reduce_mean(self.D_fake_logits) 

			self.d_loss = self.d_loss_real + self.d_loss_fake
			if self.D_loss_extra:
				self.d_loss = self.d_loss + 0.001*tf.reduce_mean(tf.square(self.D_real_logits))


		#sampler
		self.inputs_sample = tf.placeholder(
			tf.float32, [self.sample_num] + image_dims, name='real_images_sample')
		inputs_sample = self.inputs_sample

		self.sampler = G(self.z, batch_size= self.sample_num, reuse = True, layers = self.g_layers, activation = self.activation, output_dim = self.output_size,
			feature_map_shrink = self.feature_map_shrink, spatial_map_growth = self.spatial_map_growth, use_wscale = self.use_wscale, use_pixnorm = self.use_pixnorm)
		self.D_real_sample, self.D_real_logits_sample = D(inputs_sample, batch_size = self.sample_num, reuse = True, layers = self.d_layers, activation = self.activation,
		 input_dim = self.input_size, feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, z_dim = self.z_dim,
		  minibatch_std = self.minibatch_std, use_wscale = self.use_wscale)
		self.D_fake_sample, self.D_fake_logits_sample = D(self.G, batch_size = self.sample_num, reuse = True, layers = self.d_layers, activation = self.activation,
		 input_dim = self.input_size, feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, z_dim = self.z_dim,
		  minibatch_std = self.minibatch_std, use_wscale = self.use_wscale)


		if self.loss == 'RaLS':
			# d_loss
			self.d_loss_real_sample = tf.reduce_mean(
				tf.square(self.D_real_logits_sample - tf.reduce_mean(self.D_fake_logits_sample) - 1))
			self.d_loss_fake_sample = tf.reduce_mean(
				tf.square(self.D_fake_logits_sample - tf.reduce_mean(self.D_real_logits_sample) + 1))
			self.d_loss_sample = (self.d_loss_real_sample + self.d_loss_fake_sample) / 2

			if self.D_loss_extra:
				self.d_loss_sample = self.d_loss_sample + 0.001*tf.reduce_mean(tf.square(self.D_real_logits_sample))

			# g_loss
			self.g_loss_sample = (tf.reduce_mean(tf.square(self.D_fake_logits_sample - tf.reduce_mean(self.D_real_logits_sample))) / 2 +
				tf.reduce_mean(tf.square(self.D_real_logits_sample - tf.reduce_mean(self.D_fake_logits_sample))) / 2)
		elif self.loss == 'ns':
			self.d_loss_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits_sample,labels=tf.ones_like(self.D_real_logits_sample)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits_sample,labels=tf.zeros_like(self.D_fake_logits)))
			self.g_loss_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits_sample,labels=tf.ones_like(self.D_fake_logits_sample)))
			if self.D_loss_extra:
				self.d_loss_sample = self.d_loss_sample + 0.001*tf.reduce_mean(tf.square(self.D_real_logits_sample))
		elif self.loss == 'wa':
			hyperparameter = 10
			gamma = tf.random_uniform(shape=[self.sample_num,1,1,1],minval=0., maxval=1.)
			#beta = tf.ones([batch_size,1,1,1],dtype=tf.float32)
			xhat = tf.add( tf.multiply(gamma,self.inputs_sample), tf.multiply((1-gamma),self.sampler))

			_, D_xhat = D(xhat, batch_size = self.sample_num, reuse = True, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
				feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, z_dim = self.z_dim,
				 minibatch_std = self.minibatch_std, use_wscale = self.use_wscale)


			gradients = tf.gradients(D_xhat, xhat)[0]
			#print('xhatshape', xhat.shape)
			#print('idx: ', idx)
			#print('gradientdim', gradients) #(256,1,?,2) same as xhat
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
			#print('slpopedim:', slopes.shape) # (256,1)
			#gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)
			gradient_penalty = tf.reduce_mean((slopes-1.)**2)
			self.gradient_penalty = gradient_penalty
			self.d_loss_fake_sample = tf.reduce_mean(self.D_fake_logits_sample)
			self.d_loss_real_sample = -tf.reduce_mean(self.D_real_logits_sample)

			self.g_loss_sample = -tf.reduce_mean(self.D_fake_logits_sample) 
			self.d_loss_sample_wo_gp = self.d_loss_real_sample + self.d_loss_fake_sample

			self.d_loss_sample = self.d_loss_sample_wo_gp + hyperparameter*gradient_penalty
			if self.D_loss_extra:
				self.d_loss_sample = self.d_loss_sample + 0.001*tf.reduce_mean(tf.square(self.D_real_logits_sample))

		"""data visualization"""
		self.z_sum = histogram_summary("z", self.z)
		self.d_real_sum = histogram_summary("d_real", self.D_real)
		self.d_fake_sum = histogram_summary("d_fake", self.D_fake)
		#self.G_sum = image_summary("G", tf.reshape(self.G,[self.batch_size, 128, 128, 3])) #HACK
		self.G_sum = image_summary("G", self.G)
		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]
		self.saver = tf.train.Saver()

	def train(self):

		d_optim = tf.train.AdamOptimizer(
			self.learning_rate,
			beta1=self.beta1,beta2 = self.beta2, epsilon= self.epsilon).minimize(
			self.d_loss,
			var_list=self.d_vars)

		g_optim = tf.train.AdamOptimizer(
			self.learning_rate,
			beta1=self.beta1,beta2 = self.beta2, epsilon= self.epsilon).minimize(
			self.g_loss,
			var_list=self.g_vars)

		try:
			tf.global_variables_initializer().run(session=self.sess)
		except BaseException:
			tf.initialize_all_variables().run(session=self.sess)

		self.g_sum = merge_summary([self.z_sum, self.d_fake_sum,
			self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = merge_summary(
			[self.d_real_sum, self.d_loss_real_sum, self.d_loss_sum])


		if not os.path.exists('../logs/'+self.model_dir_full):
			os.makedirs('../logs/'+self.model_dir_full)
		self.writer = SummaryWriter('../logs/'+self.model_dir_full, self.sess.graph)

		if self.z_distr == 'u':
			sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim)).astype(np.float32)
			if self.normalize_z:
				sample_z /= np.sqrt(np.sum(np.square(sample_z)))
		elif self.z_distr == 'g':
			sample_z = np.random.normal(0,1,size=(self.sample_num, self.z_dim)).astype(np.float32)
			if self.normalize_z:
				sample_z /= np.sqrt(np.sum(np.square(sample_z)))
				
		# sample_z = np.full((self.sample_num, self.z_dim), 0.1).astype(np.float32) # constant input for testing


		sample_files = self.data[0:self.sample_num]
		sample = [
				get_image(
					sample_file,
					input_height=self.input_size,
					input_width=self.input_size,
					resize_height=self.output_size,
					resize_width=self.output_size,
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

		# Vectors for plotting
		gp_vec = np.array([])
		d_loss_vec = np.array([])
		d_loss_wo_gp_vec = np.array([])

		D_real_vec = np.array([])
		D_fake_vec = np.array([])

		d_loss_real_vec = np.array([])
		d_loss_fake_vec = np.array([])

		xaxis = np.array([])


		if not os.path.exists('../loss/'+self.model_dir_full):
			os.makedirs('../loss/'+self.model_dir_full)


		for epoch in xrange(self.epochs):


			np.random.shuffle(self.data)
			batch_idxs = len(self.data) // self.batch_size

			for idx in xrange(0, batch_idxs):
				batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
				batch = [
						get_image(
							batch_file,
							input_height=self.input_size,
							input_width=self.input_size,
							resize_height=self.output_size,
							resize_width=self.output_size,
							crop=self.crop) for batch_file in batch_files]
				batch_images = np.array(batch).astype(np.float32)

				if self.z_distr == 'u':
					batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
					if self.normalize_z:
						batch_z /= np.sqrt(np.sum(np.square(batch_z)))
				elif self.z_distr == 'g':
					batch_z = np.random.normal(0,1,size=(self.batch_size, self.z_dim)).astype(np.float32)
					if self.normalize_z:
						batch_z /= np.sqrt(np.sum(np.square(batch_z)))
				# batch_z = np.full((self.batch_size, self.z_dim), 0.1).astype(np.float32) # constant input for testing


				if np.mod(counter, 100) == 0:
					d_loss, D_real, D_fake, gp, d_loss_fake, d_loss_real, d_loss_wo_gp = self.sess.run(
						[self.d_loss_sample, self.D_real_sample, self.D_fake_sample, self.gradient_penalty,
						 self.d_loss_fake_sample, self.d_loss_real_sample, self.d_loss_sample_wo_gp],
						feed_dict={
							self.z: sample_z,
							self.inputs_sample: sample_inputs
						},
					)
					D_real = np.mean(D_real)
					D_fake = np.mean(D_fake)

					gp_vec = np.append(gp_vec, gp)
					d_loss_vec = np.append(d_loss_vec, d_loss)
					d_loss_wo_gp_vec = np.append(d_loss_wo_gp_vec, d_loss_wo_gp)

					D_real_vec = np.append(D_real_vec, D_real)
					D_fake_vec = np.append(D_fake_vec, D_fake)

					d_loss_real_vec = np.append(d_loss_real_vec, d_loss_real)
					d_loss_fake_vec = np.append(d_loss_fake_vec, d_loss_fake)

					xaxis = np.append(xaxis, counter*self.batch_size)

					if np.mod(counter, 10000) == 0:	
						plt.figure()
						plt.grid(True)
						plt.xlabel('Real Examples Shown')
						plt.ylabel('Loss')
						plt.title('Discriminator Loss')
						plt.tight_layout()
						plt.plot(xaxis, gp_vec,label = 'gp')
						plt.plot(xaxis, d_loss_vec,label = 'total')
						plt.plot(xaxis, d_loss_wo_gp_vec, label = '-real+fake')
						plt.legend()

						plt.savefig('../loss/' + self.model_dir_full + '/d_loss_' + str(counter*self.batch_size) +'.png')
						plt.close()

						plt.figure()
						plt.grid(True)
						plt.xlabel('Real Examples Shown')
						plt.ylabel('Loss')
						plt.title('Discriminator Probabilities on Real and Fake Data')
						plt.tight_layout()
						plt.plot(xaxis, D_real_vec,label = 'real')
						plt.plot(xaxis, D_fake_vec,label = 'fake')
						plt.legend()

						plt.savefig('../loss/' + self.model_dir_full + '/d_prob_' + str(counter*self.batch_size) +'.png')
						plt.close()

						plt.figure()
						plt.grid(True)
						plt.xlabel('Real Examples Shown')
						plt.ylabel('Loss')
						plt.title('Discriminator Loss Values on Real and Fake Data')
						plt.tight_layout()
						plt.plot(xaxis, d_loss_real_vec,label = '-real')
						plt.plot(xaxis, d_loss_fake_vec,label = 'fake')
						plt.legend()

						plt.savefig('../loss/' + self.model_dir_full + '/d_loss_val_' + str(counter*self.batch_size) +'.png')
						plt.close()



				if np.mod(counter, 1000) == 0 or np.mod(counter,1000) == 1: # CHANGE 5 to 1000 later
					samples = self.sess.run(
						self.sampler,
						feed_dict={
							self.z: sample_z,
							self.inputs_sample: sample_inputs
						},
					)
					if not os.path.exists('../{}/{}'.format('train_samples', self.model_dir_full)):
						os.makedirs('../{}/{}'.format('train_samples', self.model_dir_full))

					save_images(
						samples, 
							[int(np.sqrt(self.sample_num)),int(np.sqrt(self.sample_num))], '../{}/{}/train_{:02d}_{:04d}.png'.format(
							'train_samples', self.model_dir_full, epoch+25, idx))


				# Update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
					self.inputs: batch_images, self.z: batch_z})

				self.writer.add_summary(summary_str, counter)

				# Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
					self.inputs: batch_images, self.z: batch_z})

				self.writer.add_summary(summary_str, counter)


				if np.mod(counter, 5000) == 0:
					self.save(counter)
				counter += 1


	def save(self, step):
		model_name = "model"

		if not os.path.exists('../models/'+self.model_dir_full):
			os.makedirs('../models/'+self.model_dir_full)

		self.saver.save(self.sess, '../models/' + self.model_dir_full + '/' + model_name, global_step=step)

	def load(self):
		import re
		print(" [*] Reading models...")

		ckpt = tf.train.get_checkpoint_state('../models/'+self.model_dir_full)
		print('../models/'+self.model_dir_full)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			print(ckpt_name)
			self.saver.restore(self.sess, os.path.join('../models/', self.model_dir_full, ckpt_name))
			counter = int(
				next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find the model")
			return False, 0