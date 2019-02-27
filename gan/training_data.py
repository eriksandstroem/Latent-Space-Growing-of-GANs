
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
	swissroll, _ = sklearn.datasets.make_swiss_roll(n,noise)
	swissroll = np.asarray(swissroll)

	x = swissroll[:,0]
	y = swissroll[:,2]
	data[:,0] = x
	data[:,1] = y
	return np.array(data)

def sample_data_spiral(n=10000, noise = 0.1):
	t = np.random.uniform(0.0,9.42, n)
	x = np.multiply(t,np.cos(t)) + np.random.normal(0, noise, n)
	y = np.multiply(t,np.sin(t)) + np.random.normal(0, noise, n)
	z = t + np.random.normal(0, noise, n)

	data = np.zeros((n,3))
	data[:,0] = x
	data[:,1] = y
	data[:,2] = z
	return data