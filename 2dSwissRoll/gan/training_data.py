
import numpy as np
import math
import sklearn.datasets
import math


def get_y(x):
    return 10 + x*x


def sample_data_quad(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])
    return np.array(data)

def sample_data_swissroll(n=10000, noise = 0.1):
	data = np.zeros((n,2))
	swissroll, _ = sklearn.datasets.make_swiss_roll(n,noise) # this gives biased sampling along the line
	swissroll = np.asarray(swissroll)

	x = swissroll[:,0]
	y = swissroll[:,2]
	data[:,0] = x
	data[:,1] = y
	return np.array(data)

def sample_data_spiral(n=10000, noise = 0.1):
	t = np.random.uniform(0.0,9.42, n) # this gives an unbalanced sampling along the line
	x = np.multiply(t,np.cos(t)) + np.random.normal(0, noise, n)
	y = np.multiply(t,np.sin(t)) + np.random.normal(0, noise, n)
	z = t + np.random.normal(0, noise, n)

	data = np.zeros((n,3))
	data[:,0] = x
	data[:,1] = y
	data[:,2] = z
	return data

def sample_Z(batchsize, dim, z):
    if z == 'u':
        return np.random.uniform(-1., 1., size=[batchsize, dim])
    elif z == 'g':
        return np.random.normal(0, 1, size=[batchsize, dim])


def sample_data_sinus_swissroll(n=10000, noise = 0.5, arch = 'single'):
	data = np.zeros((n,2))

	t = np.random.uniform(4.71,14.14,n) # this gives unbalances/biased sampling along the line

	# failed Hennings sinus version
	# noise = np.absolute(amplitude*np.sin(frequency*t*(0.1*t+1)))
	# x = np.multiply(t,np.cos(t))+np.random.normal(0, noise, n)
	# y = np.multiply(t,np.sin(t))+np.random.normal(0, noise, n)

	# my versions
	if arch == 'double':
		x = np.multiply(t,np.cos(t))+np.sin(10*t)+np.random.normal(0, noise, n)
		y = np.multiply(t,np.sin(t))+np.cos(10*t)+np.random.normal(0, noise, n)
	elif arch == 'single':
		x = np.multiply(t,np.cos(t))+np.cos(10*t)+np.random.normal(0, noise, n)
		y = np.multiply(t,np.sin(t))+np.random.normal(0, noise, n)

	data[:,0] = x
	data[:,1] = y
	return np.array(data)