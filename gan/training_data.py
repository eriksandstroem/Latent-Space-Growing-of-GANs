
import numpy as np
import math
import sklearn.datasets


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