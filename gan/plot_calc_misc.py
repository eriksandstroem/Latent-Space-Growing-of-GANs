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
parser.add_argument('--n', '--noise', default=0.0, type=float, help='noise std (default: 0.0)')
parser.add_argument('--batchSize', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--i', '--iterations', default=30000, type=int, help='iterations (default: 10 000)')
parser.add_argument('--z', '--zdistribution', default='Uniform', choices=['Uniform', 'Gaussian'], help="z-distribution (default: Uniform)")
parser.add_argument('--zdim', '--zdimension', default=2, type=int, choices=[1, 2], help="z-dimension (default: 2)")
#parser.add_argument('--m', '--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
#parser.add_argument('--w', '--weight-decay', default=0, type=float, help='regularization weight decay (default: 0.0)')

# notation: a.b = #hidden layers.#neurons per layer
parser.add_argument('--g', '--generator', default='2.16', choices=['2.16', '3.2','3.8'], help="generator (default: 2.16)")
parser.add_argument('--d', '--discriminator', default='2.16', choices=['2.16', '3.2', '3.8'], help="discriminator (default: 2.16)")
parser.add_argument('--l', '--loss', default='wgan', choices=['nonsatgan', 'wgan'], help="loss function (default: 2.16)")
arg = parser.parse_args()

# create model directory to store/load old model
if not os.path.exists('../loss_plots/g_'+arg.g+'/d_'+arg.d):
    os.makedirs('../loss_plots/g_'+arg.g+'/d_'+arg.d)
if not os.path.exists('../dense_plots/g_'+arg.g+'/d_'+arg.d):
    os.makedirs('../dense_plots/g_'+arg.g+'/d_'+arg.d)

# retrieve data from log file
location = '../logs/g_'+arg.g+'/d_'+arg.d+'/logfile_noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'.log'
f  = open(location, "r")
x = f.readlines()
d_loss = np.zeros(len(x)-10)
g_loss = np.zeros(len(x)-10)
for idx, line in enumerate(x[10:]):
	split = line.split(' ')
	d = split[-1]
	g = split[-3]
	d_loss[idx] = float(d[5:])
	g_loss[idx] = float(g[5:-1])



xaxis = np.arange(0,(len(x)-10)*50,50)
# plot data
plt.figure()
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Discriminator and Generator Loss')
plt.tight_layout()
plt.plot(xaxis, d_loss,label = 'Discriminator Loss')
plt.plot(xaxis, g_loss,label = 'Generator Loss')
plt.legend()

plt.savefig('../loss_plots/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'.png')
plt.close()


# -------------- calculate discriminator statistics for generated images and real images --------------

# create the graph
def generator(Z,reuse=False):
    if arg.g == '2.16':
        with tf.variable_scope("GAN/Generator",reuse=reuse):
            h1 = tf.layers.dense(Z,16,activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,16,activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h2,2)

        return out
    elif arg.g == '3.2':
        with tf.variable_scope("GAN/Generator",reuse=reuse):
            h1 = tf.layers.dense(Z,2,activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,2,activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,2,activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h3,2)

        return out

    elif arg.g == '3.8':
        with tf.variable_scope("GAN/Generator",reuse=reuse):
            h1 = tf.layers.dense(Z,8,activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,8,activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,8,activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h3,2)

        return out

def discriminator(X,reuse=False):
    if arg.d == '2.16':
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1 = tf.layers.dense(X,16,activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,16,activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h2,1)

        return out

    elif arg.d == '3.2':
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1 = tf.layers.dense(X,2,activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,2,activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,2,activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h3,1)

        return out

    elif arg.d == '3.8':
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1 = tf.layers.dense(X,8,activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,8,activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,8,activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h3,1)

        return out


X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,arg.zdim])

G_sample = generator(Z)
r_logits = discriminator(X)
f_logits = discriminator(G_sample,reuse=True)

# initialize all variables
init_op = tf.global_variables_initializer() # create the graph
saver = tf.train.Saver()

x_plot = sample_data_swissroll(n=500, noise = arg.n)

with tf.Session() as sess:
	location = '../models/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'/model'
	saver.restore(sess, location)

	X_batch = sample_data_swissroll(n=500, noise = arg.n)
	if arg.zdim == 2:
		Z_batch = np.mgrid[-1:1:0.01, -1:1:0.01].reshape(2,-1).T
	elif arg.zdim == 1:
		Z_batch = np.arange(-1,1.01,0.01)

	d_logits = sess.run(r_logits, feed_dict={X: X_batch})
	g_logits = sess.run(f_logits, feed_dict={Z: Z_batch})
	d_prob = 1/(1+np.exp(d_logits))
	g_prob = 1/(1+np.exp(g_logits))
	mean_prob_d = np.mean(d_prob)
	mean_prob_g = np.mean(g_prob)
	#print("Average discriminator output for REAL samples:", mean_prob_d)
	#print("Average discriminator output for FAKE samples:", mean_prob_g)

	# -------------- generate points sampled densly in a grid and plot results together with discriminator score as background --------------

	g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})

	fig, (ax1, ax2, cax) = plt.subplots(ncols=3, figsize=(20,5))
	fig.subplots_adjust(wspace=0.2)
	ax1.grid(True)
	x1 = np.reshape(x_plot[:,0],(500,1))
	x2 = np.reshape(x_plot[:,1],(500,1))
	g1 = np.reshape(g_plot[:,0],(len(g_plot),1))
	g2 = np.reshape(g_plot[:,1],(len(g_plot),1))
	cd = np.reshape(d_prob,(500,1))
	cg = np.reshape(g_prob,(len(g_plot),1))
	xax = ax1.scatter(x1,x2, c = cd, marker='x')
	gax = ax1.scatter(g1, g2, c = cg, marker='.',s=2)
	ax1.legend((xax,gax), ("Real Data", "Generated Data"))
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_title('Sample Space')

	z1 = np.reshape(Z_batch[:,0],(len(Z_batch),1))
	z2 = np.reshape(Z_batch[:,1],(len(Z_batch),1))
	ax2.scatter(z1, z2, c = cg, marker='.',s=1)
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_title('Latent Space')
	fig.suptitle('Dense 2D Uniform Sampling', fontsize=16)

	loc = plticker.MultipleLocator(base=1.0)
	ax2.xaxis.set_major_locator(loc)
	ax2.yaxis.set_major_locator(loc)

	ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
	cax.set_axes_locator(ip)

	fig.colorbar(gax, cax=cax, ax=[ax1,ax2])

	textstr = 'D_r avg:'+str(mean_prob_d)[:-4]+'    D_f avg:'+str(mean_prob_g)[:-4]
	plt.text(0.30, 0.0, textstr, fontsize=14, transform=plt.gcf().transFigure)
	fig.subplots_adjust(left=0.3, bottom=0.13)
	fig.savefig('../dense_plots/g_'+arg.g+'/d_'+arg.d+'/noise_'+str(arg.n)+'_lr_'+str(arg.lr)+'_zdim_'+str(arg.zdim)+'_z_'+arg.z+'_loss_'+arg.l+'.png', bbox_inches='tight')


