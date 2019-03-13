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
parser.add_argument('--batchSize', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--n', '--noise', default=0.0, type=float, help='noise std (default: 0.0)')
parser.add_argument('--i', '--iterations', default=30000, type=int, help='iterations (default: 30 000)')
parser.add_argument('--z', '--zdistribution', default='u', choices=['u', 'g'], help="z-distribution (default: u)")
parser.add_argument('--opt', '--optimizer', default='sgd', choices=['sgd', 'rms', 'ad'], help="optimizer (default: sgd)")
parser.add_argument('--zdim', '--zdimension', default=2, type=int, choices=[1, 2], help="z-dimension (default: 2)")
#parser.add_argument('--m', '--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
#parser.add_argument('--w', '--weight-decay', default=0, type=float, help='regularization weight decay (default: 0.0)')

# notation: a.b = #hidden layers.#neurons per layer
parser.add_argument('--arch', '--architecture', default='2.16', help="architecture (default: 2.16)")
parser.add_argument('--l', '--loss', default='wa', choices=['ns', 'wa'], help="loss function (default: wa)")
arg = parser.parse_args()

# create model directory to store/load old model
if not os.path.exists('../loss_plots/'+arg.arch):
    os.makedirs('../loss_plots/'+arg.arch)
if not os.path.exists('../grid_plots/'+arg.arch):
    os.makedirs('../grid_plots/'+arg.arch)

# retrieve data from log file
location = '../logs/'+arg.arch+'/log_n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_I'+arg.init+'.log'
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

plt.savefig('../loss_plots/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_I'+arg.init+'.png')
plt.close()


# -------------- calculate discriminator statistics for generated images and real images --------------

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

X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,arg.zdim])

G_sample = generator(Z)
r_logits = discriminator(X)
f_logits = discriminator(G_sample,reuse=True)

# initialize all variables
init_op = tf.global_variables_initializer() # create the graph
saver = tf.train.Saver() #pass list of old parameters in the parentethis later

x_plot = sample_data_swissroll(n=500, noise = arg.n)

with tf.Session() as sess:
	location = '../models/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_I'+arg.init+'/model'
	#sess.run(init_op)
	saver.restore(sess, location)

	X_batch = sample_data_swissroll(n=500, noise = arg.n)
	if arg.zdim == 2:
		Z_batch = np.mgrid[-1:1:0.01, -1:1:0.01].reshape(2,-1).T
	elif arg.zdim == 1:
		Z_batch = np.arange(-1,1.01,0.01)
		Z_batch = np.reshape(Z_batch,(len(Z_batch),1))

	d_logits = sess.run(r_logits, feed_dict={X: X_batch})
	g_logits = sess.run(f_logits, feed_dict={Z: Z_batch})
	d_prob = 1/(1+np.exp(-d_logits))
	g_prob = 1/(1+np.exp(-g_logits))
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
	if arg.zdim == 2:
		gax = ax1.scatter(g1, g2, c = cg, marker='.',s=2)
	elif arg.zdim == 1:
		gax = ax1.scatter(g1, g2, c = cg, marker='.',s=20)
		
	xax = ax1.scatter(x1,x2, c = cd, marker='x')
	ax1.legend((xax,gax), ("Real Data", "Generated Data"))
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_title('Sample Space')

	if arg.zdim == 2:
		z1 = np.reshape(Z_batch[:,0],(len(Z_batch),1))
		z2 = np.reshape(Z_batch[:,1],(len(Z_batch),1))
		ax2.scatter(z1, z2, c = cg, marker='.',s=1)
	elif arg.zdim == 1:
		z1 = Z_batch
		ax2.scatter(z1, np.zeros(len(Z_batch)) ,c = cg, marker = '.', s=30)

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
	fig.savefig('../grid_plots/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_I'+arg.init+'.png', bbox_inches='tight')


