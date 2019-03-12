
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from training_data import *
import sklearn.datasets as skdata
import numpy as np
import os
import argparse
import tensorflow as tf
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


# -------------- plot loss function of discriminator and generator --------------
parser = argparse.ArgumentParser(description='Plot Loss function')
parser.add_argument('--gpu', default=1, type=int, help='epochs (default: 1)')
parser.add_argument('--n', '--noise', default=0.0, type=float, help='noise std (default: 0.0)')
parser.add_argument('--batchSize', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--i', '--iterations', default=30000, type=int, help='iterations (default: 10 000)')
parser.add_argument('--opt', '--optimizer', default='SGD', choices=['SGD', 'RMSProp', 'Adam'], help="optimizer (default: SGD)")
parser.add_argument('--z', '--zdistribution', default='Uniform', choices=['Uniform', 'Gaussian'], help="z-distribution (default: Uniform)")
parser.add_argument('--zdim', '--zdimension', default=2, type=int, choices=[1, 2], help="z-dimension (default: 2)")
parser.add_argument('--a', '--activation', default='leaky_relu', help="activation (default: Leaky relu)")
#parser.add_argument('--m', '--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
#parser.add_argument('--w', '--weight-decay', default=0, type=float, help='regularization weight decay (default: 0.0)')

# notation: a.b = #hidden layers.#neurons per layer
parser.add_argument('--g', '--generator', default='2.16', help="generator (default: 2.16)")
parser.add_argument('--d', '--discriminator', default='2.16', help="discriminator (default: 2.16)")
parser.add_argument('--l', '--loss', default='wgan', choices=['nonsatgan', 'wgan'], help="loss function (default: 2.16)")
arg = parser.parse_args()

# create model directory to store/load old model
if not os.path.exists('../models/g_'+arg.g+'grown/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'_opt_'+arg.opt):
    os.makedirs('../models/g_'+arg.g+'grown/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'_opt_'+arg.opt)
if not os.path.exists('../logs/g_'+arg.g+'grown/d_'+arg.d):
    os.makedirs('../logs/g_'+arg.g+'grown/d_'+arg.d)
if not os.path.exists('../plots/g_'+arg.g+'grown/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'_opt_'+arg.opt):
    os.makedirs('../plots/g_'+arg.g+'grown/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'_opt_'+arg.opt)

# define batchsize
batch_size = arg.batchSize

# define the graph
def generator(Z,reuse=False, arch = '2.16'):
    arch = arch.split('.')
    layers = int(arch[0])
    nodes = int(arch[1])
    with tf.variable_scope("GAN/Generator", reuse = reuse):
        for i in range(layers):
            h = tf.layers.dense(Z,nodes,activation=tf.nn.leaky_relu, name = 'h'+str(i+1))
            Z = h
        out = tf.layers.dense(h,2, name ='out')
    return out

def discriminator(X,reuse=False, arch = '2.16'):
    arch = arch.split('.')
    layers = int(arch[0])
    nodes = int(arch[1])
    with tf.variable_scope("GAN/Discriminator", reuse = reuse):
        for i in range(layers):
            h = tf.layers.dense(X, nodes, activation=tf.nn.leaky_relu, name = 'h'+str(i+1))
            X = h
        out = tf.layers.dense(h,1, name = 'out')
    return out

# --- retrieve weights form first layer in old generator ---
Z = tf.placeholder(tf.float32,[None,1])
G_sample = generator(Z, arch = arg.g)

Z_batch = np.ones((1,1)) #np.random.uniform(-1., 1., size=[1, 1]) REMOVE LATER
old_model_location = '../models/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'_opt_'+arg.opt+'/model'
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator"))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, old_model_location) 
    biash1 = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h1/bias:0")) 
    kernelh1 = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h1/kernel:0"))
    print('GAN/Generator/h1/bias:0 :', biash1) # REMOVE LATER
    print('GAN/Generator/h1/kernel:0 :', kernelh1) # REMOVE LATER
    biash2 = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h2/bias:0")) # REMOVE LATER
    kernelh2 = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h2/kernel:0")) # REMOVE LATER
    print('GAN/Generator/h2/bias:0 :', biash2) # REMOVE LATER
    print('GAN/Generator/h2/kernel:0 :', kernelh2) # REMOVE LATER
    print('G_sample: ', sess.run(G_sample, {Z : Z_batch})) # REMOVE LATER

# create extended graph
tf.reset_default_graph()









# if arg.l == 'nonsatgan':
#     disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.zeros_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))
#     gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
# elif arg.l == 'wgan':
#     hyperparameter = 10
#     alpha = tf.random_uniform(shape=[batch_size,1,1,1],minval=0., maxval=1.)
#     #alpha = tf.ones([batch_size,1,1,1],dtype=tf.float32)
#     xhat = tf.add( tf.multiply(alpha,X), tf.multiply((1-alpha),G_sample))
#     D_xhat = discriminator(xhat, reuse=True)

#     gradients = tf.gradients(D_xhat, xhat)[0]
#     #print('xhatshape', xhat.shape)
#     #print('idx: ', idx)
#     #print('gradientdim', gradients) #(256,1,?,2) same as xhat
#     slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
#     #print('slpopedim:', slopes.shape) # (256,1)
#     gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)

#     D_loss_fake = tf.reduce_mean(f_logits)
#     D_loss_real = -tf.reduce_mean(r_logits) + hyperparameter*gradient_penalty

#     gen_loss = -tf.reduce_mean(f_logits) 

#     disc_loss = D_loss_real + D_loss_fake

# gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
# disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

# tf.set_random_seed(42)
# #init_op = tf.global_variables_initializer()
# #sess = tf.Session()
# #sess.run(init_op)
# #print(sess.run(gen_vars))
# #sess.close()

# # RMSProp
# gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
# disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step
# # SGD
# #gen_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(gen_loss,var_list = gen_vars) # G Train step
# #disc_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(disc_loss,var_list = disc_vars) # D Train step
# # Adam
# #gen_step = tf.train.AdamOptimizer(learning_rate=arg.lr, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars) # G Train step
# #disc_step = tf.train.AdamOptimizer(learning_rate=arg.lr, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=disc_vars) # D Train step

# # initialize all variables
# init_op = tf.global_variables_initializer()

# # create list of old parameters
# model_location = '../models/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'/model' 
# restored_vars  = get_tensors_in_checkpoint_file(file_name=model_location)
# tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
# print('restored_vars:', restored_vars)
# print('tensors_to_load:', tensors_to_load)
# # pass list of old parameters
# saver = tf.train.Saver(tensors_to_load)

# nd_steps = 10
# ng_steps = 10
# noise = arg.n

# x_plot = sample_data_swissroll(n=arg.batchSize, noise = arg.n)

# config = tf.ConfigProto(device_count = {'GPU': arg.gpu+1})

# with tf.Session(config = config) as sess:
# 	#location = '../models/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'/model'
# 	print('before:', sess.run(init_op))
# 	saver.restore(sess, model_location)
# 	print('after:', sess.run(init_op))

# 	for i in range(arg.i):
# 	    X_batch = sample_data_swissroll(n=arg.batchSize, noise = noise)
# 	    Z_batch = sample_Z(batch_size, arg.zdim, arg.z)

# 	    for _ in range(nd_steps):
# 	        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

# 	    for _ in range(ng_steps):
# 	        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

	    
# 	    if i%50 == 0:
# 	        logger.info('==>>> iteration:{}, g loss:{}, d loss:{}'.format(i, gloss, dloss))

# 	    if i%1000 == 0:
# 	        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})

# 	        plt.figure()
# 	        plt.grid(True)
# 	        xax = plt.scatter(x_plot[:,0],x_plot[:,1])
# 	        gax = plt.scatter(g_plot[:,0], g_plot[:,1])
# 	        plt.legend((xax,gax), ("Real Data", "Generated Data"))
# 	        plt.xlabel('x')
# 	        plt.ylabel('y')
# 	        plt.title('Swiss Roll Data')
# 	        plt.tight_layout()
# 	        plt.savefig('../plots/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'/iteration_%i.png'%i)
# 	        plt.close()



# 	save_path = saver.save(sess, '../models/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'/model')
