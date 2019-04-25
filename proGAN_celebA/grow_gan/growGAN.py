import os
import scipy.misc
import numpy as np
from glob import glob

from subGAN import subGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf


class growGAN(object):
	def __init__(
			self,
			z_dims,
			epochs,
			g_layers,
			d_layers,
			output_dims,
			useAlpha,
			useBeta,
			feature_map_shrink,
			feature_map_growth,
			spatial_map_shrink,
			spatial_map_growth,
			stage,
			loss,
			z_distr,
			activation,
			weight_init,
			lr,
			beta1,
			beta2,
			epsilon,
			batch_size,
			sample_num,
			gpu,
			g_batchnorm,
			d_batchnorm,
			normalize_z,
			crop,
			trainflag,
			visualize,
			model_dir,
			minibatch_std,
			use_wscale,
			use_pixnorm,
			D_loss_extra,
			G_run_avg):

		self.z_dims = z_dims
		self.epochs = epochs
		self.g_layers = g_layers
		self.d_layers = d_layers
		self.output_dims = output_dims
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
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.sample_num = sample_num
		self.gpu = gpu
		self.g_batchnorm = g_batchnorm
		self.d_batchnorm = d_batchnorm
		self.normalize_z = normalize_z
		self.crop = crop
		self.trainflag = trainflag
		self.visualize = visualize
		self.model_dir = model_dir
		self.minibatch_std = minibatch_std
		self.use_wscale = use_wscale
		self.use_pixnorm = use_pixnorm
		self.D_loss_extra = D_loss_extra
		self.G_run_avg = G_run_avg

	def train(self):
		self.z_dims = self.z_dims.split('.')
		self.z_dims = list(map(int, self.z_dims))
		self.epochs = self.epochs.split('.')
		self.epochs = list(map(int, self.epochs))
		self.g_layers = self.g_layers.split('.')
		self.g_layers = list(map(int, self.g_layers))
		self.d_layers = self.d_layers.split('.')
		self.d_layers = list(map(int, self.d_layers))
		self.output_dims = self.output_dims.split('.')
		self.output_dims = list(map(int, self.output_dims))
		self.stage = self.stage.split('.')
		self.useAlpha = self.useAlpha.split('.')
		self.useBeta = self.useBeta.split('.')

		nbrCycles = len(self.z_dims)

		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
		run_config = tf.ConfigProto()
		run_config.gpu_options.allow_growth = True
		run_config.gpu_options.visible_device_list=str(self.gpu)
		for i in range(nbrCycles):
			print('cycle start: ', i+1)

			if i >= 1:
				oldSpecs = {
					"z_dim" : self.z_dims[i-1],
					"g_layers" : self.g_layers[i-1],
					"d_layers" : self.d_layers[i-1],
					"feature_map_shrink" : self.feature_map_shrink,
					"feature_map_growth" : self.feature_map_growth,
					"spatial_map_shrink" : self.spatial_map_shrink,
					"spatial_map_growth" : self.spatial_map_growth,
					"output_dims" : self.output_dims[i-1],
					"stage" : self.stage[i-1],
					"useAlpha" : self.useAlpha[i-1],
					"useBeta" : self.useBeta[i-1]
				}
			else:
				oldSpecs = {}

			with tf.Session(config=run_config) as sess: 
				subgan = subGAN(
					sess = sess,
					z_dim = self.z_dims[i],
					epochs = self.epochs[i],
					g_layers = self.g_layers[i],
					d_layers = self.d_layers[i],
					useAlpha = self.useAlpha[i],
					useBeta = self.useBeta[i],
					feature_map_shrink = self.feature_map_shrink,
					feature_map_growth = self.feature_map_growth,
					spatial_map_shrink = self.spatial_map_shrink,
					spatial_map_growth = self.spatial_map_growth,
					stage = self.stage[i],
					loss = self.loss,
					z_distr = self.z_distr,
					activation = self.activation,
					weight_init = self.weight_init,
					lr = self.lr,
					beta1 = self.beta1,
					beta2 = self.beta2,
					epsilon = self.epsilon,
					batch_size = self.batch_size,
					sample_num = self.sample_num,
					input_size = 128,
					output_size = self.output_dims[i],
					g_batchnorm = self.g_batchnorm,
					d_batchnorm = self.d_batchnorm,
					normalize_z = self.normalize_z,
					crop = self.crop,
					visualize = self.visualize,
					model_dir = self.model_dir,
					minibatch_std = self.minibatch_std,
					use_wscale = self.use_wscale,
					use_pixnorm = self.use_pixnorm,
					D_loss_extra = self.D_loss_extra,
					G_run_avg = self.G_run_avg,
					oldSpecs = oldSpecs)


				if self.trainflag:
					subgan.train()
					print('done training cycle: ', i+1)
				else:
					if not subgan.load()[0]: # make sure the last model is picked in this load function
						raise Exception("[!] Train a model first, then run test mode")

			tf.reset_default_graph()
				# if self.visualize: # check what this does. Is this the test mode?
				#     visualize(sess, subgan)
