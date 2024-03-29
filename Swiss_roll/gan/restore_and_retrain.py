import tensorflow as tf
import numpy as np
from training_data import *
import seaborn as sb
import matplotlib.pyplot as plt
import time
import os
import argparse
import logging
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
plt.viridis()

# -------------- plot loss function of discriminator and generator --------------
parser = argparse.ArgumentParser(description='Plot Loss function')
parser.add_argument('--gpu', default=1, type=int, help='epochs (default: 1)')
parser.add_argument('--batchSize', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--tn', '--train_noise', default=0.0, choices=[0.0, 0.5], type=float, help='training noise std (default: 0.0)')
parser.add_argument('--ptn', '--pretrain_noise', default=0.0, choices=[0.0, 0.5], type=float, help='pretrained noise (default: 0.0)')
parser.add_argument('--i', '--iterations', default=10050, type=int, help='iterations (default: 10 050)')
parser.add_argument('--z', '--zdistribution', default='u', choices=['u', 'g'], help="z-distribution (default: u)")
parser.add_argument('--opt', '--optimizer', default='sgd', choices=['sgd', 'rms', 'ad'], help="optimizer (default: sgd)")
parser.add_argument('--zdim', '--zdimension', default=2, type=int, choices=[1, 2], help="z-dimension (default: 2)")
#parser.add_argument('--m', '--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
#parser.add_argument('--w', '--weight-decay', default=0, type=float, help='regularization weight decay (default: 0.0)')

# notation: a.b = #hidden layers.#neurons per layerasp
parser.add_argument('--arch', '--architecture', default='2.16', help="architecture (default: 2.16)")
parser.add_argument('--l', '--loss', default='wa', choices=['ns', 'wa'], help="loss function (default: wa)")
parser.add_argument('--init', '--initialization', default='z', choices=['z', 'n', 'u','x'], help="growth initialization (default: z)")
parser.add_argument('--d', '--dataset', default='standard', choices=['standard', 'sinus_single', 'sinus_double'], help="dataset (default: standard)")
parser.add_argument('--advplot', '--advanced_plot', default='advanced', choices=['standard', 'advanced'], help="advanced plotting flag (default: standard)")
parser.add_argument('--w', '--wiggle_weights', default='no', choices=['no', 'yes'], help="wiggle weights flag (default: no)")
parser.add_argument('--wiggle', '--wiggle_noise', default=0.01, type=float, help='wiggle std (default: 0.01)')
parser.add_argument('--a', '--activation', default='lre', help="activation (default: leaky relu)")
arg = parser.parse_args()

# create model directory to store/load old model
if not os.path.exists('../../models/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)):
    os.makedirs('../../models/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn))
if not os.path.exists('../../logs/'+arg.d+'/'+arg.arch+'_grown'):
    os.makedirs('../../logs/'+arg.d+'/'+arg.arch+'_grown')
if not os.path.exists('../../plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)):
    os.makedirs('../../plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn))

if arg.advplot == 'advanced':
    if not os.path.exists('../../grid_plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)):
        os.makedirs('../../grid_plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn))

# Logger Setting
logger = logging.getLogger('netlog')
logger.setLevel(logging.INFO)
ch = logging.FileHandler('../../logs/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)+'.log')
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("================================================")
logger.info("Learning Rate: {}".format(arg.lr))
logger.info("Train Noise: {}".format(arg.tn))
logger.info("Pre-Train Noise: {}".format(arg.ptn))
logger.info("Iterations: {}".format(arg.i))
logger.info("Architecture: "+arg.arch)
logger.info("Batch Size: {}".format(arg.batchSize))
logger.info("Z-dimension: {}".format(arg.zdim))
logger.info("Z-distribution: {}".format(arg.z))
logger.info("Loss: "+arg.l)
logger.info("Optimizer: "+arg.opt)
logger.info("Growth Initializer: "+arg.init)
logger.info("Activation Function: "+arg.a)


# The config for GPU usage
config = tf.ConfigProto()
config.gpu_options.visible_device_list=str(arg.gpu)

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

# retrieve weights form first layer in old generator
Z = tf.placeholder(tf.float32,[None,1])
G_sample = generator(Z, arch = arg.arch)
# Z_batch = np.ones((1,1)) #np.random.uniform(-1., 1., size=[1, 1]) REMOVE LATER

old_model_location = '../../models/'+arg.d+'/'+arg.arch+'/TN'+str(arg.ptn)+'_Lr'+str(arg.lr)+'_D'+str(1)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'/model'
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
X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])
G_sample = generator(Z, arch = arg.arch)
r_logits = discriminator(X, arch = arg.arch)
f_logits = discriminator(G_sample,reuse=True, arch = arg.arch)

# specify loss function
if arg.l == 'ns':
    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))
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
    gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2)

    D_loss_fake = tf.reduce_mean(f_logits)
    D_loss_real = -tf.reduce_mean(r_logits) + hyperparameter*gradient_penalty

    gen_loss = -tf.reduce_mean(f_logits) 

    disc_loss = D_loss_real + D_loss_fake

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")


# Z_batch = np.array([[1,1]]) #np.random.uniform(-1., 1., size=[1, 2]) # REMOVE LATER

# specify optimizer
if arg.opt == 'rms':
    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
    disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step
if arg.opt == 'sgd':
    gen_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(gen_loss,var_list = gen_vars) # G Train step
    disc_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(disc_loss,var_list = disc_vars) # D Train step
if arg.opt == 'ad':
    gen_step = tf.train.AdamOptimizer(learning_rate=arg.lr, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars) # G Train step
    disc_step = tf.train.AdamOptimizer(learning_rate=arg.lr, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=disc_vars) # D Train step


reader = tf.train.NewCheckpointReader(old_model_location)
# create dictionary to restore all weights but the first layer weights
restore_dict = dict()
for v in tf.trainable_variables():
    tensor_name = v.name.split(':')[0]
    print('tensor name: ', tensor_name)
    if reader.has_tensor(tensor_name) and 'Generator/h1' not in tensor_name:
        print('to restore: yes')
        restore_dict[tensor_name] = v
    else:
        print('to restore: no')

nd_steps = 10
ng_steps = 10

x_plot = sample_data_swissroll(n=arg.batchSize, noise = arg.tn)

# architecture
arch = arg.arch.split('.')
layers = int(arch[0])
nodes = int(arch[1])

with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(restore_dict)
    saver.restore(sess, old_model_location)
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

    if arg.w == 'yes':
        for key in restore_dict:
            split = key.split('/')
            if split[1] != 'Discriminator':
                #print('key: ', key) # REMOVE LATER
                tensor = tf.get_default_graph().get_tensor_by_name(key + ':0')
                #print('before: ', sess.run(tensor)) # REMOVE LATER
                assign_op = tf.assign(tensor, tensor + np.squeeze(np.random.normal(0,arg.wiggle, np.shape(tensor))))
                sess.run(assign_op)
                #print('after: ', sess.run(tensor)) # REMOVE LATER
    saver = tf.train.Saver()
    # print('GAN/Generator/h1/bias:0 new:', sess.run(biash1)) # REMOVE LATER
    # print('GAN/Generator/h1/kernel:0 new:', sess.run(kernelh1)) # REMOVE LATER
    # biash2 = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h2/bias:0")) # REMOVE LATER
    # kernelh2 = sess.run(tf.get_default_graph().get_tensor_by_name("GAN/Generator/h2/kernel:0")) # REMOVE LATER
    # print('GAN/Generator/h2/bias:0 new:', biash2) # REMOVE LATER
    # print('GAN/Generator/h2/kernel:0 new:', kernelh2) # REMOVE LATER
    # print('G_sample new: ', sess.run([G_sample], feed_dict={Z: Z_batch})) # REMOVE LATER

    for i in range(arg.i):
        if arg.d == 'standard':
            X_batch = sample_data_swissroll(n=arg.batchSize, noise = arg.tn)
        elif arg.d == 'sinus_single':
            X_batch = sample_data_sinus_swissroll(n=arg.batchSize, noise = arg.tn, arch = 'single')
        elif arg.d == 'sinus_double':
            X_batch = sample_data_sinus_swissroll(n=arg.batchSize, noise = arg.tn, arch = 'double')
        Z_batch = sample_Z(arg.batchSize, arg.zdim, arg.z)

        if arg.advplot == 'standard' and i%1000 == 0:
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
            if arg.w == 'no':
                plt.savefig('../../plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)+'/iteration_%i.png'%i)
            elif arg.w == 'yes':
                plt.savefig('../../plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)+'/iteration_%i'+'_w'+str(wiggle_std)+'.png'%i)
            plt.close()

        if arg.advplot == 'advanced' and i%100 == 0:
            if arg.zdim == 2:
                Z_dense = np.mgrid[-1:1:0.01, -1:1:0.01].reshape(2,-1).T
            elif arg.zdim == 1:
                Z_dense = np.arange(-1,1.01,0.0001)
                Z_dense = np.reshape(Z_batch_dense,(len(Z_dense),1))
            X_dense = np.mgrid[-15:15.06:0.05, -15:15.06:0.05].reshape(2,-1).T
            d_logits_dense = sess.run(r_logits, feed_dict={X: X_dense})
            d_logits = sess.run(r_logits, feed_dict={X: X_batch})
            g_logits = sess.run(f_logits, feed_dict={Z: Z_dense})
            print('iteration: ', i)
            print('max: ', max(d_logits_dense))
            print('min: ', min(d_logits_dense))
            d_prob_dense = 1/(1+np.exp(-d_logits_dense))
            d_prob = 1/(1+np.exp(-d_logits))
            g_prob = 1/(1+np.exp(-g_logits))
            mean_prob_d = np.mean(d_prob)
            mean_prob_g = np.mean(g_prob)

            g_plot = sess.run(G_sample, feed_dict={Z: Z_dense})
            x_plot = X_batch

            fig, (ax0, ax1, ax2, cax) = plt.subplots(ncols=4, figsize=(40,7))
            fig.subplots_adjust(wspace=0.2)
            ax1.grid(True)
            x1 = np.reshape(x_plot[:,0],(arg.batchSize,1))
            x2 = np.reshape(x_plot[:,1],(arg.batchSize,1))
            g1 = np.reshape(g_plot[:,0],(len(g_plot),1))
            g2 = np.reshape(g_plot[:,1],(len(g_plot),1))
            cd = np.reshape(d_prob,(arg.batchSize,1))
            cg = np.reshape(g_prob,(len(g_plot),1))
            cd_dense = np.reshape(d_prob_dense,(len(d_prob_dense),1))
            if arg.zdim == 2:
                gax = ax1.scatter(g1, g2, c = cg, marker='.',s=2)
            elif arg.zdim == 1:
                gax = ax1.scatter(g1, g2, c = cg, marker='.',s=20)
                
            # xax = ax1.scatter(x1,x2, c = cd, marker='x')
            xax = ax1.scatter(x1,x2, marker='x')
            ax1.legend((xax,gax), ("Real Data", "Generated Data"))
            ax1.set_title('Sample Space')

            if arg.zdim == 2:
                z1 = np.reshape(Z_dense[:,0],(len(Z_dense),1))
                z2 = np.reshape(Z_dense[:,1],(len(Z_dense),1))
                ax2.scatter(z1, z2, c = cg, marker='.',s=10)
            elif arg.zdim == 1:
                z1 = Z_dense
                ax2.scatter(z1, np.zeros(len(Z_dense)) ,c = cg, marker = '.', s=30)

            ax2.set_title('Latent Space')

            loc = plticker.MultipleLocator(base=1.0)
            ax2.xaxis.set_major_locator(loc)
            ax2.yaxis.set_major_locator(loc)

            ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
            cax.set_axes_locator(ip)

            x1_dense = np.reshape(X_dense[:,0],(len(X_dense),1))
            x2_dense = np.reshape(X_dense[:,1],(len(X_dense),1))
            ax0.scatter(x1_dense, x2_dense, c = cd_dense, marker='.',s=1)

            ax0.set_title('Sample Space')
            fig.suptitle('Dense 2D Uniform Sampling', fontsize=16)

            ax0.xaxis.set_ticks(np.arange(-15,15,5))
            ax0.yaxis.set_ticks(np.arange(-15,15,5))

            fig.colorbar(gax, cax=cax, ax=[ax0,ax1,ax2])

            textstr = 'D_r avg:'+str(mean_prob_d)[:-4]+'    D_f avg:'+str(mean_prob_g)[:-4]
            plt.text(0.30, 0.0, textstr, fontsize=14, transform=plt.gcf().transFigure)
            fig.subplots_adjust(left=0.3, bottom=0.13, wspace = 0.2)
            
            I = str(i)
            if arg.w == 'no':
                fig.savefig('../../grid_plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+
                    '_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)+'/iteration_'+I+'test.png', bbox_inches='tight')
            elif arg.w == 'yes':
                fig.savefig('../../grid_plots/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+
                    '_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)+'/iteration_'+I+'_w'+str(arg.wiggle)+'.png', bbox_inches='tight')

        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})


        if i%50 == 0:
            logger.info('==>>> iteration:{}, g loss:{}, d loss:{}'.format(i, gloss, dloss))

        if i%1000 == 0:
            saver.save(sess, '../../models/'+arg.d+'/'+arg.arch+'_grown/TN'+str(arg.tn)+'_Lr'+str(arg.lr)+'_D'+str(arg.zdim)+'_Z'+arg.z+'_L'+arg.l+'_OP'+arg.opt+'_ACT'+arg.a+'_I'+arg.init+'_PTN'+str(arg.ptn)+'/model'+str(i))

