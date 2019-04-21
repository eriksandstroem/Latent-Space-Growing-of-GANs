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

class subGAN(object):
	def __init__(
		self,
		sess,
		z_dim,
		epochs,
		g_layers,
		d_layers,
		useAlpha,
		useBeta,
		feature_map_shrink,
		feature_map_growth,
		spatial_map_shrink,
		spatial_map_growth,
		stage,
		loss,
		z_distr, # NOT TAKEN CARE OF
		activation,
		weight_init,  # NOT TAKEN CARE OF
		lr,
		beta1,
		beta2,
		epsilon,
		batch_size,
		sample_num,
		input_size,
		output_size,
		g_batchnorm,
		d_batchnorm,
		normalize_z,  # NOT TAKEN CARE OF
		crop,
		visualize,  # NOT TAKEN CARE OF
		model_dir,
		oldSpecs):

		self.sess = sess
		self.z_dim = z_dim
		self.epochs = epochs
		self.g_layers = g_layers
		self.d_layers = d_layers
		self.useAlpha = useAlpha
		self.useBeta = useBeta
		self.feature_map_shrink = feature_map_shrink
		self.feature_map_growth = feature_map_growth
		self.spatial_map_shrink = spatial_map_shrink
		self.spatial_map_growth = spatial_map_growth
		self.stage = stage
		self.loss = loss
		self.z_distr = z_distr
		self.activation = activation
		self.weight_init = weight_init
		self.learning_rate = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.sample_num = sample_num
		self.input_size = 128
		self.output_size = output_size
		self.g_batchnorm = g_batchnorm
		self.d_batchnorm = d_batchnorm
		self.normalize_z = normalize_z
		self.crop = crop
		self.visualize = visualize
		self.model_dir = model_dir
		self.model_dir_full = model_dir +  '/stage_'+self.stage+'_z'+str(self.z_dim)
		self.oldSpecs = oldSpecs

		self.data = glob(os.path.join("../../celebA_dataset", 'celebA', '*.jpg'))
		self.data.sort()
		seed = 547
		np.random.seed(seed)
		#np.random.shuffle(self.data)

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
		self.alpha = tf.placeholder(tf.float32, shape=(), name = 'alpha')
		self.beta = tf.placeholder(tf.float32, shape=(), name = 'beta')

		self.G = G(self.z, batch_size= self.batch_size, reuse = False, bn = self.g_batchnorm, layers = self.g_layers, activation = self.activation, output_dim = self.output_size,
			feature_map_shrink = self.feature_map_shrink, spatial_map_growth = self.spatial_map_growth, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta)
		self.D_real, self.D_real_logits = D(inputs, batch_size = self.batch_size, reuse = False, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta, z_dim = self.z_dim)
		self.D_fake, self.D_fake_logits = D(self.G, batch_size = self.batch_size, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta, z_dim = self.z_dim)


		"""loss function"""
		if self.loss == 'RaLS':
			# d_loss
			self.d_loss_real = tf.reduce_mean(
			    tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits) - 1))
			self.d_loss_fake = tf.reduce_mean(
			    tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits) + 1))
			self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2

			# g_loss
			self.g_loss = (tf.reduce_mean(tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits))) / 2 
				+ tf.reduce_mean(tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits))) / 2)
		elif self.loss == 'ns':
			self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits,labels=tf.ones_like(self.D_real_logits)))
			self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,labels=tf.zeros_like(self.D_fake_logits)))
			self.d_loss = self.d_loss_real + self.d_loss_fake
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,labels=tf.ones_like(self.D_fake_logits)))
		elif self.loss == 'wa':
			hyperparameter = 10
			gamma = tf.random_uniform(shape=[self.batch_size,1,1,1],minval=0., maxval=1.)
			#beta = tf.ones([batch_size,1,1,1],dtype=tf.float32)
			xhat = tf.add( tf.multiply(gamma,self.inputs), tf.multiply((1-gamma),self.G))

			_, D_xhat = D(xhat, batch_size = self.batch_size, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta, z_dim = self.z_dim)

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


		#sampler
		self.inputs_sample = tf.placeholder(
			tf.float32, [self.sample_num] + image_dims, name='real_images_sample')
		inputs_sample = self.inputs_sample

		self.sampler = G(self.z, batch_size= self.sample_num, reuse = True, bn = self.g_batchnorm, layers = self.g_layers, activation = self.activation, output_dim = self.output_size,
			feature_map_shrink = self.feature_map_shrink, spatial_map_growth = self.spatial_map_growth, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta)
		self.D_real_sample, self.D_real_logits_sample = D(inputs_sample, batch_size = self.sample_num, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta, z_dim = self.z_dim)
		self.D_fake_sample, self.D_fake_logits_sample = D(self.G, batch_size = self.sample_num, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta, z_dim = self.z_dim)


		if self.loss == 'RaLS':
			# d_loss
			self.d_loss_real_sample = tf.reduce_mean(
				tf.square(self.D_real_logits_sample - tf.reduce_mean(self.D_fake_logits_sample) - 1))
			self.d_loss_fake_sample = tf.reduce_mean(
				tf.square(self.D_fake_logits_sample - tf.reduce_mean(self.D_real_logits_sample) + 1))
			self.d_loss_sample = (self.d_loss_real_sample + self.d_loss_fake_sample) / 2

			# g_loss
			self.g_loss_sample = (tf.reduce_mean(tf.square(self.D_fake_logits_sample - tf.reduce_mean(self.D_real_logits_sample))) / 2 +
				tf.reduce_mean(tf.square(self.D_real_logits_sample - tf.reduce_mean(self.D_fake_logits_sample))) / 2)
		elif self.loss == 'ns':
			self.d_loss_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits_sample,labels=tf.ones_like(self.D_real_logits_sample)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits_sample,labels=tf.zeros_like(self.D_fake_logits)))
			self.g_loss_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits_sample,labels=tf.ones_like(self.D_fake_logits_sample)))
		elif self.loss == 'wa':
			hyperparameter = 10
			gamma = tf.random_uniform(shape=[self.sample_num,1,1,1],minval=0., maxval=1.)
			#beta = tf.ones([batch_size,1,1,1],dtype=tf.float32)
			xhat = tf.add( tf.multiply(gamma,self.inputs_sample), tf.multiply((1-gamma),self.sampler))

			_, D_xhat = D(xhat, batch_size = self.sample_num, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
				feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha, useAlpha = self.useAlpha, beta = self.beta, useBeta = self.useBeta, z_dim = self.z_dim)


			gradients = tf.gradients(D_xhat, xhat)[0]
			#print('xhatshape', xhat.shape)
			#print('idx: ', idx)
			#print('gradientdim', gradients) #(256,1,?,2) same as xhat
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
			#print('slpopedim:', slopes.shape) # (256,1)
			#gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)
			gradient_penalty = tf.reduce_mean((slopes-1.)**2)
			D_loss_fake = tf.reduce_mean(self.D_fake_logits_sample)
			D_loss_real = -tf.reduce_mean(self.D_real_logits_sample) + hyperparameter*gradient_penalty

			self.g_loss_sample = -tf.reduce_mean(self.D_fake_logits_sample) 

			self.d_loss_sample = D_loss_real + D_loss_fake

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

		#sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
		sample_z = np.full((self.sample_num, self.z_dim), 0.1).astype(np.float32)

		alpha = np.float32(0.0)
		beta = np.float32(0.0)

		sample_files = self.data[0:self.sample_num]
		if self.useAlpha == 'n':
			sample = [
				get_image(
					sample_file,
					input_height=self.input_size,
					input_width=self.input_size,
					resize_height=self.output_size,
					resize_width=self.output_size,
					crop=self.crop) for sample_file in sample_files]
		elif self.useAlpha == 'y':
			sample = [
			get_image_interpolate(
				sample_file,
				input_height=self.input_size,
				input_width=self.input_size,
				resize_height=self.output_size,
				resize_width=self.output_size,
				crop=self.crop, alpha = 1.0) for sample_file in sample_files]

		sample_inputs = np.array(sample).astype(np.float32)

		counter = 0
		start_time = time.time()

		could_load, model_counter, message = self.load(weight_init = self.weight_init)
		if could_load:
			counter = model_counter
			print(" [*] " + message)
		else:
			print(" [!] " + message)

		for epoch in xrange(1):


			# np.random.shuffle(self.data)
			batch_idxs = len(self.data) // self.batch_size

			for idx in xrange(0, batch_idxs):
				batch_files = self.data[0 * self.batch_size:(0 + 1) * self.batch_size] # replace 0 with idx
				if self.useAlpha == 'n':
					batch = [
						get_image(
							batch_file,
							input_height=self.input_size,
							input_width=self.input_size,
							resize_height=self.output_size,
							resize_width=self.output_size,
							crop=self.crop) for batch_file in batch_files]
				elif self.useAlpha == 'y':
					batch = [
						get_image_interpolate(
							batch_file,
							input_height=self.input_size,
							input_width=self.input_size,
							resize_height=self.output_size,
							resize_width=self.output_size,
							crop=self.crop, alpha = alpha) for batch_file in batch_files]
				batch_images = np.array(batch).astype(np.float32)

                #batch_images_shape = batch_images.shape #HACK
                #batch_images = np.reshape(batch_images, [batch_images_shape[0], batch_images_shape[3], batch_images_shape[1], batch_images_shape[2]]) #HACK

				# batch_z = np.random.uniform(-1, 1,[self.batch_size, self.z_dim]).astype(np.float32)
				batch_z = np.full((self.batch_size, self.z_dim), 0.1).astype(np.float32)



				# # Run g_optim twice to make sure that d_loss does not go to
				# # zero (different from paper)
				# _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
				# 	self.inputs: batch_images, self.z: batch_z})

				# self.writer.add_summary(summary_str, counter)

				errD_fake = self.d_loss_fake.eval(
					{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
				errD_real = self.d_loss_real.eval(
					{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
				errG = self.g_loss.eval(
					{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
				d_real_logits = self.D_real_logits.eval({self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
				d_fake_logits = self.D_fake_logits.eval({self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
				g_out = self.G.eval({self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
                
				print(
					"Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
						(epoch, idx, batch_idxs, time.time() - start_time, errD_real, errG)) # errD_fake + errD_real
				print('g_out: ', g_out)# errD_fake + errD_real
				print('d_fake_logits: ', d_fake_logits)
				print('d_real_logits: ', d_real_logits)
				print('errD_fake: ', errD_fake)
				

				# # Update D network
				# _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
				# 	self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})

				# self.writer.add_summary(summary_str, counter)

				# # Update G network
				# _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
				# 	self.z: batch_z, self.alpha: alpha, self.beta: beta})

				# self.writer.add_summary(summary_str, counter)

				if np.mod(counter, 5) == 0:
					samples, d_loss, g_loss, D_real, D_fake = self.sess.run(
						[self.sampler, self.d_loss_sample, self.g_loss_sample, self.D_real_sample, self.D_fake_sample],
						feed_dict={
							self.z: sample_z,
							self.inputs_sample: sample_inputs, self.alpha: alpha, self.beta: beta
						},
					)
					if not os.path.exists('../{}/{}'.format('train_samples', self.model_dir_full)):
						os.makedirs('../{}/{}'.format('train_samples', self.model_dir_full))

					save_images(
						samples, 
							[int(np.sqrt(self.sample_num)),int(np.sqrt(self.sample_num))], '../{}/{}/train_{:02d}_{:04d}.png'.format(
							'train_samples', self.model_dir_full, epoch, idx))
					print("[Sample] d_loss: %.8f, g_loss: %.8f" %
						(d_loss, g_loss))
				# if np.mod(counter, 5) == 0:
					# self.save(counter)

				# Update D network
				# _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
				# 	self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})

				# self.writer.add_summary(summary_str, counter)

				# # Update G network
				# _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
				# 	self.z: batch_z, self.alpha: alpha, self.beta: beta})

				# self.writer.add_summary(summary_str, counter)

				counter += 1
				if alpha < 1:
					alpha = alpha + 0.5
					print('alpha: ', alpha)
				if beta < 1:
					beta = beta + 0.5
				if idx == 2:  # REMOVE LATER!!!
					alpha = 1.0
					beta = 1.0
					errD_fake = self.d_loss_fake.eval(
						{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
					errD_real = self.d_loss_real.eval(
						{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
					errG = self.g_loss.eval(
						{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha, self.beta: beta})
					d_real_logits = self.D_real_logits.eval({self.inputs: batch_images, self.alpha: alpha, self.beta: beta})
					d_fake_logits = self.sess.run(self.D_fake_logits, feed_dict={self.z: batch_z, self.alpha: alpha, self.beta: beta})
					g_out = self.sess.run(self.G, feed_dict={self.z: batch_z, self.alpha: alpha, self.beta: beta})
	                
					print(
						"Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
							(epoch, idx, batch_idxs, time.time() - start_time, errD_real, errG)) # errD_fake + errD_real
					print('d_fake_logits: ', d_fake_logits)
					print('d_real_logits: ', d_real_logits)
					print('errD_fake: ', errD_fake)
					samples, d_loss, g_loss, D_real, D_fake = self.sess.run(
						[self.sampler, self.d_loss_sample, self.g_loss_sample, self.D_real_sample, self.D_fake_sample],
						feed_dict={
							self.z: sample_z,
							self.inputs_sample: sample_inputs, self.alpha: alpha, self.beta: beta
						},
					)
					print("[Sample] d_loss: %.8f, g_loss: %.8f" %
						(d_loss, g_loss))
					g_out = self.sess.run(self.G, feed_dict={self.z: batch_z, self.alpha: alpha, self.beta: beta})
					print(g_out)
					self.save(counter)
					break

    # @property
    # def model_dir(self):
    #     return "Arch{}_Zd{}_L{}_Bs{}_Lr{}_Zd{}_Iwh{}_Owh{}_Bn{}_classic_hopefix".format(
    #         self.architecture, self.z_dim, self.loss, self.batch_size, self.learning_rate, self.zdistribution,
    #         self.input_height, self.output_width, str(self.batchnorm))

	def save(self, step):
		model_name = "model"

		if not os.path.exists('../models/'+self.model_dir_full):
			os.makedirs('../models/'+self.model_dir_full)

		self.saver.save(self.sess, '../models/' + self.model_dir_full + '/' + model_name, global_step=step)

	# def load(self):
	# 	import re
	# 	print(" [*] Reading models...")

	# 	ckpt = tf.train.get_checkpoint_state('../models/'+self.model_dir)
	# 	if ckpt and ckpt.model_checkpoint_path:
	# 		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	# 		self.saver.restore(self.sess, os.path.join('../models',
	# 			self.model_dir, ckpt_name))
	# 		counter = int(
	# 			next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
	# 		print(" [*] Success to read {}".format(ckpt_name))
	# 		return True, counter
	# 	else:
	# 		print(" [*] Failed to find the model")
	# 		return False, 0

	def load(self, weight_init = 'z'):
		import re
		print(" [*] Reading models...")
		try:
			zList = os.listdir('../models/'+self.model_dir)
		except:
			variables = tf.trainable_variables()
			for v in variables:
				tensor_name = v.name.split(':')[0]
				# print('tensor name: ', tensor_name)
			return False, 0, 'First Training Cycle'
	
		old_model_location = '../models/'+self.model_dir+'/stage_'+self.oldSpecs["stage"]+'_z'+str(self.oldSpecs['z_dim'])
		ckpt = tf.train.get_checkpoint_state(old_model_location)

		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		old_model_location = old_model_location + '/' + ckpt_name

		if self.stage == 'i':
			# restore the old layers and add two new layers in generator and discriminator
			reader = tf.train.NewCheckpointReader(old_model_location)
			# I CAN USE THIS METHOD WHEN I KNOW WHAT LAYERS CONTAIN THE SAME WEIGHTS, BUT FIRST I NEED TO FIND THAT SOMEHOW
			restore_dict = dict()
			variables = tf.trainable_variables()
			discVar = tf.trainable_variables(scope = 'discriminator')
			for v in variables:
				tensor_name = v.name.split(':')[0]
				# print('tensor name: ', tensor_name)
				if reader.has_tensor(tensor_name) and 'generator' in tensor_name:
					restore_dict[tensor_name] = v
					# print('exists in old generator')
				elif reader.has_tensor(tensor_name) and ('out' in tensor_name or 'h1/' in tensor_name):
					restore_dict[tensor_name] = v
					# print('exists in old discriminator chill')
				elif reader.has_tensor(tensor_name):
					name_in_graph = tensor_name.split('/')
					#print(name_in_graph)
					name = name_in_graph[1]
					#print(name)
					if name[-2].isdigit():
						nbr = int(name[-2:])+2
						name = name[:-2]+str(nbr)
					else:
						nbr = int(name[-1])+2
						name = name[:-1]+str(nbr)
					name_in_graph[1] = name
					# print(name)
					s = '/'
					name_in_graph = s.join(name_in_graph)
					# print(name_in_graph)
					for var in discVar:
						varname = var.name.split(':')[0]
						# print(varname)
						if varname == name_in_graph:
							var_restore = var
							break
					restore_dict[tensor_name] = var_restore
					# print('exists in old discriminator')

			# print('done')
			saver = tf.train.Saver(restore_dict)
			saver.restore(self.sess, old_model_location)

			return True, 0, 'Added two more layers and restored the old layers, doubling the output dimension'
		elif self.stage == 'f':
			# same number of layers as before, but now we extend the number of channels for the layers
			# we are already in a session. We can easily create a dictionary and restore in the generator layer h4 and the final conv1x1. They have the same names as in the previous network.
			# In the discriminator I can restore h1, h2, h3 easily. We have also already initialized all weights previously, which is nice. The next problem is how we restore the partially new layers
			# In order to do that we need to extract the useful feature maps from the previous network checkpoint and use the assign function for the new network. 

			# Restore the not grown layers in discriminator and generator
			reader = tf.train.NewCheckpointReader(old_model_location)
			restore_dict = dict()
			partial_restore_dict = dict()
			for v in tf.trainable_variables():
				tensor_name = v.name.split(':')[0]
				# print('tensor name: ', tensor_name)
				name = tensor_name.split('/')
				name = name[1]
				if name[-2].isdigit():
					nbr = int(name[-2:])
				elif name[-1].isdigit():
					nbr = int(name[-1])

				if reader.has_tensor(tensor_name) and 'generator/g_out' in tensor_name:
					restore_dict[tensor_name] = v
					# print('JAtensor name: ', tensor_name)
				elif reader.has_tensor(tensor_name) and 'generator' in tensor_name and nbr == self.g_layers:
					restore_dict[tensor_name] = v
					# print('JAtensor name: ', tensor_name)
				elif reader.has_tensor(tensor_name) and 'discriminator/d_h1/' in tensor_name or 'discriminator/d_h2' in tensor_name: # or 'discriminator/d_h3' in tensor_name:
					restore_dict[tensor_name] = v
					# print('JAtensor name: ', tensor_name)
				else:
					partial_restore_dict[tensor_name] = reader.get_tensor(tensor_name)

			saver = tf.train.Saver(restore_dict)
			saver.restore(self.sess, old_model_location)

			for tensor_name, tensorValue in partial_restore_dict.items():
				# retrieve tensor from current graph
				print('tensorname: ', tensor_name)
				tensor = tf.get_default_graph().get_tensor_by_name(tensor_name+':0')
				print('oldtensorshape: ', tensorValue.shape)
				print('newtensorshape: ', tensor.shape)
				tensor_name_split = tensor_name.split('/')
				if tensor_name_split[-1] == 'bias':
					tensorValue = tensorValue.astype(np.float32)
					assign_op = tf.assign(tensor, tensorValue) # why don't we have this in the dictionary directly?
					# self.sess.run(assign_op)
				elif tensor_name_split[-1] == 'biases':	
					if tensorValue.shape == tensor.shape:
						tensorValue = tensorValue.astype(np.float32)
						assign_op = tf.assign(tensor, tensorValue)  # why don't we have this in the dictionary directly?
						# self.sess.run(assign_op)
					else:
						assign_op = tf.assign(tensor, np.append(tensorValue, np.zeros((1,tensorValue.shape[0]))))
						# self.sess.run(assign_op)
				elif tensor_name_split[-1] == 'kernel':
					if tensorValue.shape[1] != 1:
						filterSize = tensorValue.shape[1]
						maps = tensorValue.shape[0]
						channels = maps
						imSize = tensorValue.shape[1]/channels
						w = int(np.sqrt(imSize))
						h = w
						tensorValue = np.reshape(tensorValue, [maps, w, h, channels])
						temp = np.concatenate((tensorValue, np.random.normal(0,1,(tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 3)
						temp = np.concatenate((temp,np.random.normal(0,1,(temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3]))),axis = 0)
						temp = np.reshape(temp, [maps*2,filterSize*2])
						temp = temp.astype(np.float32)
						assign_op = tf.assign(tensor, temp)
						# self.sess.run(assign_op)
					else: 
						temp = np.concatenate((tensorValue, np.random.normal(0,1,(tensorValue.shape[0],tensorValue.shape[1]))), axis = 0)
						temp = temp.astype(np.float32)
						assign_op = tf.assign(tensor, temp)
						# self.sess.run(assign_op)
				elif tensor_name_split[-1] == 'w':
					if tensorValue.shape[2] == tensor.shape[2] and tensorValue.shape[3] == tensor.shape[3]:
						tensorValue = tensorValue.astype(np.float32)
						assign_op = tf.assign(tensor, tensorValue)
					# double fourth axis
					elif tensorValue.shape[2] == tensor.shape[2]:
						temp = np.concatenate((tensorValue,np.random.normal(0,1,(tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 3)
						# temp = np.concatenate((np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3])),np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 3)
						temp = temp.astype(np.float32)
						assign_op = tf.assign(tensor, temp)
						# self.sess.run(assign_op)
					# double third axis
					elif tensorValue.shape[3] == tensor.shape[3]:
						temp = np.concatenate((tensorValue,np.random.normal(0,1,(tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 2)
						# temp = np.concatenate((np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3])), tensorValue), axis = 2)
						# temp = np.concatenate((np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3])),np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 2)
						temp = temp.astype(np.float32)
						assign_op = tf.assign(tensor, temp)
						# self.sess.run(assign_op)
					# double both third and fourth axis
					else:
						temp = np.concatenate((tensorValue,np.random.normal(0,1,(tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 3)
						temp = np.concatenate((temp,np.random.normal(0,1,(temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3]))), axis = 2)
						# temp = np.concatenate((np.zeros((temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3])), temp), axis = 2)
						# temp = np.concatenate((np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3])),np.zeros((tensorValue.shape[0],tensorValue.shape[1],tensorValue.shape[2],tensorValue.shape[3]))), axis = 3)
						# temp = np.concatenate((temp,np.zeros((temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3]))), axis = 2)
						temp = temp.astype(np.float32)
						assign_op = tf.assign(tensor, temp)
				
				self.sess.run(assign_op)


			return True, 0, 'Added feature channels and initialized the new channels with '+weight_init+', doubled the latent space dimension'
