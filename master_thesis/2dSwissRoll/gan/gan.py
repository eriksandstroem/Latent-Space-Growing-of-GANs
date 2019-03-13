import tensorflow as tf
import numpy as np
from training_data import *
import seaborn as sb
import matplotlib.pyplot as plt
import time
import os
import argparse
import logging
import matplotlib
matplotlib.use('agg')
plt.ioff()

parser = argparse.ArgumentParser(description='Tensorflow Training')
parser.add_argument('--gpu', default=1, type=int, help='epochs (default: 1)')
parser.add_argument('--batchSize', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--n', '--noise', default=0.0, type=float, help='noise std (default: 0.0)')
parser.add_argument('--i', '--iterations', default=30050, type=int, help='iterations (default: 30 050)')
parser.add_argument('--z', '--zdistribution', default='u', choices=['u', 'g'], help="z-distribution (default: u)")
parser.add_argument('--opt', '--optimizer', default='sgd', choices=['sgd', 'rms', 'ad'], help="optimizer (default: sgd)")
parser.add_argument('--zdim', '--zdimension', default=2, type=int, choices=[1, 2], help="z-dimension (default: 2)")
#parser.add_argument('--m', '--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
#parser.add_argument('--w', '--weight-decay', default=0, type=float, help='regularization weight decay (default: 0.0)')

# notation: a.b = #hidden layers.#neurons per layer
parser.add_argument('--arch', '--architecture', default='2.16', help="architecture (default: 2.16)")
parser.add_argument('--l', '--loss', default='wa', choices=['ns', 'wa'], help="loss function (default: wa)")
parser.add_argument('--a', '--activation', default='lre', help="activation (default: leaky relu)")
arg = parser.parse_args()

# create model directory to store/load old model
if not os.path.exists('../../models/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a):
    os.makedirs('../../models/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a)
if not os.path.exists('../../logs/'+arg.arch):
    os.makedirs('../../logs/'+arg.arch)
if not os.path.exists('../../plots/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a):
    os.makedirs('../../plots/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a)

# Logger Setting
logger = logging.getLogger('netlog')
logger.setLevel(logging.INFO)
ch = logging.FileHandler('../../logs/'+arg.arch+'/log_n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'.log')
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("================================================")
logger.info("Learning Rate: {}".format(arg.lr))
logger.info("Noise: {}".format(arg.n))
logger.info("Iterations: {}".format(arg.i))
logger.info("Architecture: "+arg.arch)
logger.info("Batch Size: {}".format(arg.batchSize))
logger.info("Z-dimension: {}".format(arg.zdim))
logger.info("Z-distribution: {}".format(arg.z))
logger.info("Loss: "+arg.l)
logger.info("Optimizer: "+arg.opt)
logger.info("Activation Function: "+arg.a)

sb.set()

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
# def generator(Z,reuse=False):
#     if arg.g == '2.16':
#         with tf.variable_scope("GAN/Generator",reuse=reuse):
#             h1 = tf.layers.dense(Z,16,activation=tf.nn.leaky_relu)
#             h2 = tf.layers.dense(h1,16,activation=tf.nn.leaky_relu)
#             out = tf.layers.dense(h2,2)

#         return out
#     elif arg.g == '3.2':
#         with tf.variable_scope("GAN/Generator",reuse=reuse):
#             h1 = tf.layers.dense(Z,2,activation=tf.nn.leaky_relu)
#             h2 = tf.layers.dense(h1,2,activation=tf.nn.leaky_relu)
#             h3 = tf.layers.dense(h2,2,activation=tf.nn.leaky_relu)
#             out = tf.layers.dense(h3,2)

#         return out

#     elif arg.g == '3.8':
#         with tf.variable_scope("GAN/Generator",reuse=reuse):
#             h1 = tf.layers.dense(Z,8,activation=tf.nn.leaky_relu)
#             h2 = tf.layers.dense(h1,8,activation=tf.nn.leaky_relu)
#             h3 = tf.layers.dense(h2,8,activation=tf.nn.leaky_relu)
#             out = tf.layers.dense(h3,2)

#         return out
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
# def discriminator(X,reuse=False):
#     if arg.d == '2.16':
#         with tf.variable_scope("GAN/Discriminator",reuse=reuse):
#             h1 = tf.layers.dense(X,16,activation=tf.nn.leaky_relu)
#             h2 = tf.layers.dense(h1,16,activation=tf.nn.leaky_relu)
#             out = tf.layers.dense(h2,1)

#         return out

#     elif arg.d == '3.2':
#         with tf.variable_scope("GAN/Discriminator",reuse=reuse):
#             h1 = tf.layers.dense(X,2,activation=tf.nn.leaky_relu)
#             h2 = tf.layers.dense(h1,2,activation=tf.nn.leaky_relu)
#             h3 = tf.layers.dense(h2,2,activation=tf.nn.leaky_relu)
#             out = tf.layers.dense(h3,1)

#         return out

#     elif arg.d == '3.8':
#         with tf.variable_scope("GAN/Discriminator",reuse=reuse):
#             h1 = tf.layers.dense(X,8,activation=tf.nn.leaky_relu)
#             h2 = tf.layers.dense(h1,8,activation=tf.nn.leaky_relu)
#             h3 = tf.layers.dense(h2,8,activation=tf.nn.leaky_relu)
#             out = tf.layers.dense(h3,1)

#         return out


X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,arg.zdim])

G_sample = generator(Z, arch = arg.arch)
r_logits = discriminator(X, arch = arg.arch)
f_logits = discriminator(G_sample,reuse=True, arch = arg.arch)


if arg.l == 'ns':
    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.zeros_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
elif arg.l == 'wa':
    hyperparameter = 10
    alpha = tf.random_uniform(shape=[arg.batchSize,1,1,1],minval=0., maxval=1.)
    #alpha = tf.ones([batch_size,1,1,1],dtype=tf.float32)
    xhat = tf.add( tf.multiply(alpha,X), tf.multiply((1-alpha),G_sample))
    D_xhat = discriminator(xhat, reuse=True)

    gradients = tf.gradients(D_xhat, xhat)[0]
    #print('xhatshape', xhat.shape)
    #print('idx: ', idx)
    #print('gradientdim', gradients) #(256,1,?,2) same as xhat
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3]))
    #print('slpopedim:', slopes.shape) # (256,1)
    #gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    D_loss_fake = tf.reduce_mean(f_logits)
    D_loss_real = -tf.reduce_mean(r_logits) + hyperparameter*gradient_penalty

    gen_loss = -tf.reduce_mean(f_logits) 

    disc_loss = D_loss_real + D_loss_fake



gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

if arg.opt == 'rms':
    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
    disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step
if arg.opt == 'sgd':
    gen_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(gen_loss,var_list = gen_vars) # G Train step
    disc_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(disc_loss,var_list = disc_vars) # D Train step
if arg.opt == 'ad':
    gen_step = tf.train.AdamOptimizer(learning_rate=arg.lr, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars) # G Train step
    disc_step = tf.train.AdamOptimizer(learning_rate=arg.lr, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=disc_vars) # D Train step

# include saver
saver = tf.train.Saver()

config = tf.ConfigProto(device_count = {'GPU': arg.gpu+1})
# The config for GPU usage
config = tf.ConfigProto()
config.gpu_options.visible_device_list=str(arg.gpu)
sess = tf.Session(config=config)
tf.global_variables_initializer().run(session=sess)


nd_steps = 10
ng_steps = 10

x_plot = sample_data_swissroll(n=arg.batchSize, noise = arg.n)

for i in range(arg.i):
    X_batch = sample_data_swissroll(n=arg.batchSize, noise = arg.n)
    Z_batch = sample_Z(arg.batchSize, arg.zdim, arg.z)

    for _ in range(nd_steps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    
    if i%50 == 0:
        logger.info('==>>> iteration:{}, g loss:{}, d loss:{}'.format(i, gloss, dloss))

    if i%1000 == 0:
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})

        plt.figure()
        plt.grid(True)
        xax = plt.scatter(x_plot[:,0],x_plot[:,1])
        gax = plt.scatter(g_plot[:,0], g_plot[:,1])
        plt.legend((xax,gax), ("Real Data", "Generated Data"))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Swiss Roll Data')
        plt.tight_layout()
        plt.savefig('../../plots/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'/iteration_%i.png'%i)
        plt.close()



saver.save(sess, '../../models/'+arg.arch+'/n'+str(arg.n)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'/model')
sess.close()
