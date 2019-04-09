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
		self.model_dir = model_dir +  '/z'+str(self.z_dim)+'_out_'+str(self.output_size)
		self.oldSpecs = oldSpecs

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
		self.alpha = tf.placeholder(tf.float32, shape=(), name ='alpha') 

		self.G = G(self.z, batch_size= self.batch_size, reuse = False, bn = self.g_batchnorm, layers = self.g_layers, activation = self.activation, output_dim = self.output_size,
			feature_map_shrink = self.feature_map_shrink, spatial_map_growth = self.spatial_map_growth, stage = self.stage, alpha = self.alpha)
		self.D_real, self.D_real_logits = D(inputs, batch_size = self.batch_size, reuse = False, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha)
		self.D_fake, self.D_fake_logits = D(self.G, batch_size = self.batch_size, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha)


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
			beta = tf.random_uniform(shape=[self.batch_size,1,1,1],minval=0., maxval=1.)
			#beta = tf.ones([batch_size,1,1,1],dtype=tf.float32)
			xhat = tf.add( tf.multiply(beta,self.inputs), tf.multiply((1-beta),self.G))

			_, D_xhat = D(xhat, batch_size = self.batch_size, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha)

			gradients = tf.gradients(D_xhat, xhat)[0]
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
			feature_map_shrink = self.feature_map_shrink, spatial_map_growth = self.spatial_map_growth, stage = self.stage, alpha = self.alpha)
		self.D_real_sample, self.D_real_logits_sample = D(inputs_sample, batch_size = self.sample_num, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha)
		self.D_fake_sample, self.D_fake_logits_sample = D(self.G, batch_size = self.sample_num, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
			feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha)


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
			beta = tf.random_uniform(shape=[self.sample_num,1,1,1],minval=0., maxval=1.)
			#beta = tf.ones([batch_size,1,1,1],dtype=tf.float32)
			xhat = tf.add( tf.multiply(beta,self.inputs_sample), tf.multiply((1-beta),self.sampler))

			_, D_xhat = D(xhat, batch_size = self.sample_num, reuse = True, bn = self.d_batchnorm, layers = self.d_layers, activation = self.activation, input_dim = self.input_size,
				feature_map_growth = self.feature_map_growth, spatial_map_shrink = self.spatial_map_shrink, stage = self.stage, alpha = self.alpha)


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


		if not os.path.exists('../logs/'+self.model_dir):
			os.makedirs('../logs/'+self.model_dir)
		self.writer = SummaryWriter('../logs/'+self.model_dir, self.sess.graph)

		sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

		alpha = np.float32(0.0)

		sample_files = self.data[0:self.sample_num]
		if self.stage == 'f':
			sample = [
				get_image(
					sample_file,
					input_height=self.input_size,
					input_width=self.input_size,
					resize_height=self.output_size,
					resize_width=self.output_size,
					crop=self.crop) for sample_file in sample_files]
		elif self.stage == 'i':
			sample = [
			get_image_interpolate(
				sample_file,
				input_height=self.input_size,
				input_width=self.input_size,
				resize_height=self.output_size,
				resize_width=self.output_size,
				crop=self.crop, alpha = alpha) for sample_file in sample_files]

		sample_inputs = np.array(sample).astype(np.float32)

		counter = 0
		start_time = time.time()

		# RESTORE PREVIOUS MODEL IF THERE EXIST ONE
		# RESTORES CURRENT MODEL IF IT HAS BEEN INTERRUPTED AND LOADS COUNTER
		# could_load, model_counter, message = self.loadPrev(weigh_init = self.weigh_init)
	 #    if could_load:
	 #        counter = model_counter
	 #        print(" [*] " + message)
	 #    else:
	 #        print(" [!] First Training Cycle")


		# could_load, model_counter = self.load()
		# if could_load:
		# 	counter = model_counter
		# 	print(" [*] Load SUCCESS")
		# else:
		# 	print(" [!] Load failed...")

		for epoch in xrange(self.epochs):


			np.random.shuffle(self.data)

			batch_idxs = len(self.data) // self.batch_size

			for idx in xrange(0, batch_idxs):
				batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
				if self.stage == 'f':
					batch = [
						get_image(
							batch_file,
							input_height=self.input_size,
							input_width=self.input_size,
							resize_height=self.output_size,
							resize_width=self.output_size,
							crop=self.crop) for batch_file in batch_files]
				elif self.stage == 'i':
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

				batch_z = np.random.uniform(-1, 1,[self.batch_size, self.z_dim]).astype(np.float32)

				# Update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
					self.inputs: batch_images, self.z: batch_z, self.alpha: alpha})

				self.writer.add_summary(summary_str, counter)

				# Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
				self.inputs: batch_images, self.z: batch_z, self.alpha: alpha})

				self.writer.add_summary(summary_str, counter)

				# # Run g_optim twice to make sure that d_loss does not go to
				# # zero (different from paper)
				# _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
				# 	self.inputs: batch_images, self.z: batch_z})

				# self.writer.add_summary(summary_str, counter)

				errD_fake = self.d_loss_fake.eval(
					{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha})
				errD_real = self.d_loss_real.eval(
					{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha})
				errG = self.g_loss.eval(
					{self.inputs: batch_images, self.z: batch_z, self.alpha: alpha})

                
				print(
					"Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
						(epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

				if np.mod(counter, 5) == 0:
					samples, d_loss, g_loss, D_real, D_fake = self.sess.run(
						[self.sampler, self.d_loss_sample, self.g_loss_sample, self.D_real_sample, self.D_fake_sample],
						feed_dict={
							self.z: sample_z,
							self.inputs_sample: sample_inputs, self.alpha: alpha,
						},
					)
					if not os.path.exists('../{}/{}'.format('train_samples', self.model_dir)):
						os.makedirs('../{}/{}'.format('train_samples', self.model_dir))

					save_images(
						samples, 
							[int(np.sqrt(self.sample_num)),int(np.sqrt(self.sample_num))], '../{}/{}/train_{:02d}_{:04d}.png'.format(
							'train_samples', self.model_dir, epoch, idx))
					print("[Sample] d_loss: %.8f, g_loss: %.8f" %
						(d_loss, g_loss))
				if np.mod(counter, 5) == 0:
					self.save(counter)
				counter += 1
				if alpha < 1:
					alpha = alpha + float(1/batch_idxs)
					# print('alpha: ', alpha)
				if idx == 5:  # REMOVE LATER!!!
					break

    # @property
    # def model_dir(self):
    #     return "Arch{}_Zd{}_L{}_Bs{}_Lr{}_Zd{}_Iwh{}_Owh{}_Bn{}_classic_hopefix".format(
    #         self.architecture, self.z_dim, self.loss, self.batch_size, self.learning_rate, self.zdistribution,
    #         self.input_height, self.output_width, str(self.batchnorm))

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

	def loadPrev(self, weight_init = 'z'):
		import re
		print(" [*] Reading models...")
		zList = os.listdir('../models/'+self.model_dir)
		if not zList:
			return False, 0
		else:
			zList = [int(z[1:]) for z in zList]
			z = max(zList)
			path = '../model/'+self.model_dir+'/z'+str(z)
		ckpt = tf.train.get_checkpoint_state(path)

		if ckpt and ckpt.model_checkpoint_path and z == self.z_dim:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join('../models',
				self.model_dir, ckpt_name))
			counter = int(
				next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter, 'Continue Training Model with Z Dimension ' + str(z)
		else:
			# algoritmen ska vara väldigt flexibel och kunna växa från vilket nätverk som helst till ett annat där vikterna delas mellan nätverken dvs. det vore coolt om man kan träna
			# ett stort nätverk och restora ett mindre, men vet ej om det bara blir bökigt. Ska vi tex. anta att nästa modell har dubbel z-dim? inget antagande om antal lager kan göras
			# vi kan exempelvis restora hela gamla modellen och spara ner alla variabler som vi vill restora där det bara är ett subset av vikterna i ett visst lager vi vill restora.
			# om vi vill restora ett lager rakt av så går det att göra smidiagare.
			# HUR TAR JAG REDA PÅ SPECSEN AV FÖRRA NÄTVERKET? DET KÄNNS KRITISTK FÖR ATT VETA VILkA NYA VIKTER SOM SKA LÄGGA TILL OSV. är enkelt att göra genom att feeda detta då subGAN skapas.
			# rita ner ditt drömexempel på hur det ska växas och tänk utifrån det hur vi ska designa algoritmen. Drömexempel klart
			# specs: z_dims = 8.16.32.64.128.256, g:{f/n, n}, d:{n,n}, g_layers = 2.4.6.8.10.12, d_layers = 3.5.7.9.11.13, output_dims = 4.8.16.32.64.128
			# smoothness lägger jag till senare
		 	# assume the old specs are know via porting from ols subGAN. What factors need to be taken into account and in what order should I restore? We have layers, spatial stuff and feature maps. hmmm
		 	# THE IDEA: go through the layers. Of the new model. Each layer that exists (per number) in the old model) will need to be saved (the parameters) in a directory - valuie and name. The layers 
		 	# than don't exist in the old model but in the new, we only need the name maybe for, or maybe nothing for since those will just be initialized normally I guess? Or maybe to zero too. But we really need
		 	# the smooth shortcut if we initialize to zero so we get something out. If we initialize to not zero we get somethgin out of the model but it will not be the same as before growing.
		 	# Think, however, about the fact that we want to be able to grow from the same network to the same new one (technically). This means we need to go deeper. We need to find what really characterizes a
		 	# layer and compare the old layer parameters vs the new layer parameters with the same name. If the number of parameters are the same then they are identical and we will just restore the layer
		 	# in the convenient way. If the parameter numbers are different we restore according to our plan.
		 	# This way, we should end up with two directories of some sort where one is for resotring the convenient way and the other contains the values we need for restoring in a later session. But 
		 	# how can we study two graphs at the same time with names of layers that are identical? Seems impossible.MAybe the shape of the layer kernel is an easier metric to use?

		 	# CHECKING FOR ADDED
		 	# LAYERS: The name of the layer does not exist in the old model, but in the new only
		 	# CHANNELS: The shape of the layer is different between the old and new model

		 	# HOW TO DEAL WITH THE STRANGE ONES: Generator: conv1x1 and conv4x4. Discriminator: conv4x4 and FC.
		 	# Think about the fading in problem of only one layer at each growing cycle

		 	# I realize that it is too aggressive to grow two layers at once. In proGAN they only grow one layer at each iterations with a fading cycle of 4 epochs and 4 more epochs for stabilization. 
		 	# This is needed. Therefore I will need to change the dream example to account for growing only one layer at a time. This should be fine. I can just put in the generator the 1x1 conv after 
		 	# the first 3x3 conv. instead of after 2 conv3x3 like I usually do. For the discriminator we need the FC and conv4x4 as a package I believe. Draw the growing cycles again, first with fixed z_dim.
		 	# The next thing to take into account here is the growing of the z_dimension. It seems too aggressive to grow the z dimension at the same time as the layer dimension so we should probbly keep them separated
		 	# such that they can be learned in a stable manner.


# FÖR ATT FÅ TAG PÅ BIAS OCH KERNEL SOM VARiABLER
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator"))
with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, old_model_location) 
    biash1_old = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h1/bias:0")) 
    kernelh1_old = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h1/kernel:0"))
    if arg.w == 'yes':
        # print('shape biash1_old: ', np.shape(biash1_old)) # REMOVE LATER
        # print('shape kernelh1_old: ', np.shape(kernelh1_old)) # REMOVE LATER
        # print('len biash1_old: ', len(biash1_old)) # REMOVE LATER
        # print('len kernelh1_old: ', len(kernelh1_old)) # REMOVE LATER
        # print('shape normal: ', np.shape(np.random.normal(0,0.01,(1,len(biash1_old))))) # REMOVE LATER
        biash1_old = biash1_old + np.squeeze(np.random.normal(0,0.001,np.shape(biash1_old))) # fiddle around with the standard devation
        kernelh1_old = kernelh1_old + np.squeeze(np.random.normal(0,0.001,np.shape(kernelh1_old))) # fiddle around with the standard devation
    # print('GAN/Generator/h1/bias:0 old:', biash1_old) # REMOVE LATER
    # print('GAN/Generator/h1/kernel:0 old:', kernelh1_old) # REMOVE LATER
    # biash2_old = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h2/bias:0")) # REMOVE LATER
    # kernelh2_old = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h2/kernel:0")) # REMOVE LATER
    # print('GAN/Generator/h2/bias:0 old:', biash2_old) # REMOVE LATER
    # print('GAN/Generator/h2/kernel:0 old:', kernelh2_old) # REMOVE LATER
    # print('G_sample old: ', sess.run(G_sample, {Z : Z_batch})) # REMOVE LATER


# reset graph
tf.reset_default_graph()

# nätverket definieras här

reader = tf.train.NewCheckpointReader(old_model_location)
# create dictionary to restore all weights but the first layer weights
# I CAN USE THIS METHOD WHEN I KNOW WHAT LAYERS CONTAIN THE SAME WEIGHTS, BUT FIRST I NEED TO FIND THAT SOMEHOW
restore_dict = dict()
for v in tf.trainable_variables():
    tensor_name = v.name.split(':')[0]
    print('tensor name: ', tensor_name)
    if reader.has_tensor(tensor_name) and 'Generator/h1' not in tensor_name:
        print('to restore: yes')
        restore_dict[tensor_name] = v
    else:
        print('to restore: no')

# SLUNGA IN DE GAMLA VÄRDENA I DEN NYA GRAFEN.
    # architecture
arch = arg.arch.split('.')
layers = int(arch[0])
nodes = int(arch[1])

with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(restore_dict)
    saver.restore(sess, old_model_location) # RESTORA DE LAGER SOM HAR EXAKT LIKA MÅNGA VIKTER VID VÄXNINGEN
    biash1 = tf.get_default_graph().get_tensor_by_name("GAN/Generator/h1/bias:0")
    kernelh1 = tf.get_default_graph().get_tensor_by_name("GAN/Generator/h1/kernel:0")
    assign_opbias = tf.assign(biash1, biash1_old)
    if arg.init == 'z':
        assign_opkernel = tf.assign(kernelh1, np.array([np.squeeze(kernelh1_old), np.squeeze(np.zeros((1,nodes)))]))
    elif arg.init == 'n':
        assign_opkernel = tf.assign(kernelh1, np.array([np.squeeze(kernelh1_old), np.squeeze(np.random.normal(0,0.01,(1,nodes)))])) # experiment with different std
    elif arg.init == 'u':
        assign_opkernel = tf.assign(kernelh1, np.array([np.squeeze(kernelh1_old), np.squeeze(np.random.uniform(-0.1,0.1,(1,nodes)))])) # experiment with different range
    elif arg.init == 'x':
        limit = np.sqrt(6/(2+nodes))
        assign_opkernel = tf.assign(kernelh1, np.array([np.squeeze(kernelh1_old), np.squeeze(np.random.uniform(-limit,limit,(1,nodes)))]))
    sess.run(assign_opbias)   
    sess.run(assign_opbias)
    sess.run(assign_opkernel)
