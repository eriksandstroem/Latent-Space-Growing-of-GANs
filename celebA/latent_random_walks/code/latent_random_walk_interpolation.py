import tensorflow as tf
import os
import numpy as np
import sys
import scipy.misc
import copy

seed = 547
np.random.seed(seed)

arch = 'bGAN'
# arch = 'clgGAN'
arch_path = '../../'+arch+'/'
# arch = 'bgGAN'
# arch = 'lgGAN'
# arch = 'cgGAN'
model_path = '0.0001_32_44_12_13_128_nnnn_wa_g_lrelu_16_True_True_False_False_False'
# model_path = '0.0001_16_44_12_13_128_nnnn_wa_g_lrelu_16_True_True_False_False_False'
# model_path = '0.0001_8_44_12_13_128_nnnn_wa_g_lrelu_16_True_True_False_False_False'
# model_path = 'mixing_0.0001_8.16.32.64.128.256_4.8.8.8.8.8_2.4.6.8.10.12_3.5.7.9.11.13_4.8.16.32.64.128_nnnn_wa_g_lrelu_16_True_True_wobetaresg/stage_f_z256'
# model_path = 'mixing_0.0001_8.16.32.64.128.256_4.8.8.8.8.8_12.12.12.12.12.12_13.13.13.13.13.13_128.128.128.128.128.128_nnnn_wa_g_lrelu_z_16_True_True_True_True_False_False_False/stage_f_z256'
# model_path = '0.0001_256.256.256.256.256.256_4.8.8.8.8.8_2.4.6.8.10.12_3.5.7.9.11.13_4.8.16.32.64.128_nnnn_wa_g_lrelu_z_16_True_True_True_True_False_False_False/output_dim_128'
# model_path = 'mixing_0.0001_8.16.32.64.128.256_4.8.8.8.8.8_12.12.12.12.12.12_13.13.13.13.13.13_128.128.128.128.128.128_nnnn_wa_g_lrelu_z_16_True_True_True_True_False_False_False/stage_f_z256'
# model_path = '0.0001_256_44_12_13_128_nnnn_wa_g_lrelu_16_True_True_False_False_False'
full_model_path = arch_path+'models/'+model_path
sys.path.append(arch_path+'code/')

from model import *
from ops import *
from utils import *

# specs
z_dim = 32
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

    zPlaceHolder = tf.placeholder(tf.float32, [64, z_dim], name='z')
    # Generator = G(zPlaceHolder, batch_size= batch_size, reuse = False, layers = layers, activation = activation, output_dim = output_dim,
    #             feature_map_shrink = feature_map_shrink, spatial_map_growth = spatial_map_growth,
    #              beta = beta, useBeta = useBeta, alpha = alpha, useAlpha = useAlpha)

    Generator = G(zPlaceHolder, batch_size= batch_size, reuse = False, layers = layers, activation = activation, output_dim = output_dim,
                feature_map_shrink = feature_map_shrink, spatial_map_growth = spatial_map_growth,
                 use_wscale = False, use_pixnorm = False)

    a, b = load(full_model_path, session = sess)
    # saver = tf.train.Saver()
    # ckpt_name = 'model-100000'
    # saver.restore(sess, os.path.join(full_model_path, ckpt_name))
    print(a,b)
    nbrPoints = 5
    # zdimList = ['1-8_'+str(nbrPoints)+'pts','9-16_'+str(nbrPoints)+'pts','17-32_'+str(nbrPoints)+'pts','33-64_'+str(nbrPoints)+'pts','65-128_'+str(nbrPoints)+'pts','129-256_'+str(nbrPoints)+'pts']
    zdimList = ['1-8_'+str(nbrPoints)+'pts','9-16_'+str(nbrPoints)+'pts','17-32_'+str(nbrPoints)+'pts']
    # zdims = np.array([[0,8,16,32,64,128],[8,16,32,64,128,256]])
    zdims = np.array([[0,8,16],[8,16,32]])

    
    zList = []
    for i in range(nbrPoints):
        z = np.random.normal(0,1,size=(batch_size, 256)).astype(np.float32)
        z /= np.sqrt(np.sum(np.square(z)))
        z = z[:,:z_dim]
        zList.append(z)

    zList = np.array(zList)

    for i, zrange in enumerate(zdimList):
        if not os.path.exists('../{}/{}/{}'.format(arch, model_path, zrange)):
            os.makedirs('../{}/{}/{}'.format(arch, model_path, zrange))

        zListTemp = copy.copy(zList)
        zTemp = zListTemp[0]
        for r in range(nbrPoints-1):
            zListTemp[r+1,:,zdims[1][i]:] = zTemp[:,zdims[1][i]:]
            zListTemp[r+1,:,:zdims[0][i]] = zTemp[:,:zdims[0][i]]

        c = 0
        # print('-----------------------')
        for p in range(nbrPoints-1):
            z_start = zListTemp[p]
            z_end = zListTemp[p+1]
            # print('z_start: ', z_start)
            # print('z_end: ', z_end)
            # print(np.sum(z_start[:,zdims[1][i]:]-z_end[:,zdims[1][i]:]))
            # print(np.sum(z_start[:,zdims[0][i]:zdims[1][i]]-z_end[:,zdims[0][i]:zdims[1][i]]))

            steps = 25
            counter = 0
            for k in range(steps):
                sample_z = counter*z_end+(1-counter)*z_start
                # sample_z = sample_z.astype(np.float32)
                # print(sample_z.shape)

                # samples = sess.run(
                #     Generator,
                #     feed_dict={
                #         zPlaceHolder: sample_z,
                #     },
                # )
                # save_images(
                #     samples, 
                #         [int(np.sqrt(batch_size)),int(np.sqrt(batch_size))], '../{}/{}/{}/sample_{:02d}.png'.format(
                #         arch, model_path, zrange, c))
                counter += 1/steps
                c += 1


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
