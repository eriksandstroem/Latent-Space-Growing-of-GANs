import tensorflow as tf
import os
import numpy as np
import sys
import scipy.misc
import copy

seed = 547
np.random.seed(seed)

# arch = 'bGAN'
arch = 'clgGAN'
# arch = 'bgGAN'
# arch = 'lgGAN'
# arch = 'cgGAN'
arch_path = '../../'+arch+'/'
# model_path = '0.0001_256_44_12_13_128_nnnn_wa_g_lrelu_16_True_True_False_False_False'
model_path = 'mixing_0.0001_8.16.32.64.128.256_4.8.8.8.8.8_2.4.6.8.10.12_3.5.7.9.11.13_4.8.16.32.64.128_nnnn_wa_g_lrelu_16_True_True_wobetaresg/stage_f_z256'
# model_path = 'mixing_0.0001_8.16.32.64.128.256_4.8.8.8.8.8_12.12.12.12.12.12_13.13.13.13.13.13_128.128.128.128.128.128_nnnn_wa_g_lrelu_z_16_True_True_False_True_False_False_False/stage_f_z256'
# model_path = '0.0001_256.256.256.256.256.256_4.8.8.8.8.8_2.4.6.8.10.12_3.5.7.9.11.13_4.8.16.32.64.128_nnnn_wa_g_lrelu_z_16_True_True_True_True_False_False_False/output_dim_128'
# model_path = 'mixing_0.0001_8.16.32.64.128.256_4.8.8.8.8.8_12.12.12.12.12.12_13.13.13.13.13.13_128.128.128.128.128.128_nnnn_wa_g_lrelu_z_16_True_True_False_True_False_False_False/stage_f_z256'
# model_path = '0.0001_256_44_12_13_128_nnnn_wa_g_lrelu_16_True_True_False_False_False'
full_model_path = arch_path+'models/'+model_path
sys.path.append(arch_path+'code/')

from model import *
from ops import *
from utils import *

# specs
z_dim = 256
batch_size = 64
layers = 12
activation = 'lrelu'
output_dim = 128
feature_map_shrink = 'n'
spatial_map_growth = 'n'
useBeta = 'n'
useAlpha = 'n'
beta = 1
alpha = 1
gpu = 0
setting = 'normal' # ['normal', '', '']

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.visible_device_list=str(gpu)

    sess = tf.Session(config=run_config)

    z = tf.placeholder(tf.float32, [None, z_dim], name='z')
    Generator = G(z, batch_size= batch_size, reuse = False, layers = layers, activation = activation, output_dim = output_dim,
                feature_map_shrink = feature_map_shrink, spatial_map_growth = spatial_map_growth,
                 beta = beta, useBeta = useBeta, alpha = alpha, useAlpha = useAlpha)

    a, b = load(full_model_path, session = sess)
    # saver = tf.train.Saver()
    # ckpt_name = 'model-100000'
    # saver.restore(sess, os.path.join(full_model_path, ckpt_name))
    print(a,b)

    # zdimList = ['1-8','9-16','17-32','33-64','65-128','129-256']
    # zdims = np.array([[0,8,16,32,64,128],[8,16,32,64,128,256]])

    z_start = np.random.normal(0,1,size=(batch_size, 256)).astype(np.float32)
    z_start /= np.sqrt(np.sum(np.square(z_start)))

    # temp = np.full((batch_size, 248), 0.0).astype(np.float32)
    # z_start[:,8:] = temp

    # sample_z1 = sample_z1[:,0:z_dim]
    sample_z2 = np.random.normal(0,1,size=(batch_size, 256)).astype(np.float32) # Note that by sampling 64 new 256 element vectors, all images are interpolated in different directions
    sample_z2 /= np.sqrt(np.sum(np.square(sample_z2)))
    print(np.sum(np.absolute(z_start-sample_z2)))

    # for i, zrange in enumerate(zdimList):
    idxs = [1, 5, 250, 90, 195, 180, 235]
    for i in idxs:
	    if not os.path.exists('../{}/{}/{}'.format(arch, model_path, str(i))):
	        os.makedirs('../{}/{}/{}'.format(arch, model_path, str(i)))
	    # print(i)
	    # print(str(zdims[0][i])+':'+str(zdims[1][i]))
	    # z_insert = sample_z2[:,zdims[0][i]:zdims[1][i]]

	    z_aim = copy.copy(z_start)
	    a = -0.125

	    z_aim[:,i] = a
	    # z_aim[:,zdims[0][i]:zdims[1][i]] = z_insert
	    # print(np.sum(np.absolute(z_aim-z_start)))
	    # print(z_aim)
	    # print(z_start)



	    steps = 50
	    counter = 0
	    for k in range(steps):
	        z_aim[:,i] = a
	        # sample_z = counter*z_aim+(1-counter)*z_start

	        sample_z = z_aim
	        samples = sess.run(
	            Generator,
	            feed_dict={
	                z: sample_z,
	            },
	        )
	        save_images(
	            samples, 
	                [int(np.sqrt(batch_size)),int(np.sqrt(batch_size))], '../{}/{}/{}/sample_{:02d}.png'.format(
	                arch, model_path, str(i), k))
	        counter += 1/steps
	        a += 0.25/steps



def load(model_path, session):
    import re
    print(" [*] Reading models...")

    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(full_model_path, ckpt_name))
        counter = int(
            next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find the model")
        return False, 0


if __name__ == '__main__':
    tf.app.run()
