from __future__ import print_function, division

from keras.layers import Input, Dense, GaussianNoise, LeakyReLU
from keras.models import Sequential, Model

from keras.layers import Input, Dense, GaussianNoise, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import scipy.io as sio
import argparse
from math import *
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.utils import shuffle
from statsmodels.distributions.empirical_distribution import ECDF as fit_ecdf
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import train_test_split


def inverse_transform_sampling(hist, bin_edges):
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist)
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    return inv_cdf

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true* y_pred)

def reciprocal_loss(y_true, y_pred):
    return K.mean(K.pow(y_true*y_pred,-1))

def my_binary_crossentropy(y_true, y_pred):
    return -K.mean(K.log(y_true)+K.log(y_pred))

def logsumexp_loss(y_true, y_pred):
    loss = K.logsumexp(y_pred) - K.log(tf.cast(K.shape(y_true)[0], tf.float32))
    return loss

def phi(x, mu, sigma):
    N,D = np.shape(x)
    unif_output = np.zeros((N,D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output



#Maximum Mean Discrepancy (MMD)
def mmd_score(x,y, kernel="pow", sigma=1.0, beta=1.0, sigmas_mixture=[0.001, 0.01, 0.15, 0.25, 0.50, 0.75], drop_xx=False, drop_yy=False):
    """
    Compute Maximum Mean Discrepancy (MMD) between two multivariate distributions based on samples.
    For kernel = "pow" and beta=1.0 the MMD is equivalent to the energy distance.

    Args:
        x (numpy array): Samples from distribution p
        y (numpy array): Samples from distribution q
        kernel (str, optional): Type of kernel function. Defaults to "rbf".
        sigma (float, optional): bandwidth parameter for RBF kernel. Defaults to 1.0.
        beta (float, optional): Power for power kernel. Defaults to 1.0. This kernel recovers the energy distance.
        sigmas_mixture (list, optional): Bandwidths for mixture of RBF kernels. Defaults to [0.01, 0.1, 1.0, 10.0].

    Returns:
        float: MMD score
    """

    beta=1.0
    N = x.shape[0]
    M = y.shape[0]
    

    if kernel == "rbf":
        def kern(a):
            return np.exp(-np.power(a,2)/(2*sigma**2))

    elif kernel == "pow":
        def kern(a):
            return -np.power(a,beta)

    elif kernel == "rbf_mixture":
        def kern(a):
            res = 0.0
            for sig in sigmas_mixture:
                res = res + np.exp(-np.power(a,2)/(2*sig**2))
            return res

    def get_score(x,y):
        res = 0.0
        for i in range(x.shape[0]):
            res = res + np.sum(kern(np.sqrt(np.sum(np.square(x[[i],:] - y), axis=1))))
        return res

    if (drop_xx is False) and (drop_yy is False):
        return get_score(x,x)/(N*(N-1)) - 2*get_score(x,y)/(N*M)  + get_score(y,y)/(M*(M-1))

    else:
        if drop_xx:
            return get_score(y,y)/(M*(M-1)) - 2*get_score(x,y)/(N*M)
        elif drop_yy:
            return get_score(x,x)/(N*(N-1)) - 2*get_score(x,y)/(N*M)

############### define loss ################
def ed(y_data, y_model):
    """Compute Energy distance

    Args:
        y_data (tf.tensor, shape 1xDxN): Samples from true distribution.
        y_model (tf. tensor, shape 1xDxM): Samples from model.

    Returns:
        tf.float: Energy distance for batch
    """
    n_samples_model = tf.cast(tf.shape(y_model)[2], dtype=tf.float32)
    n_samples_data = tf.cast(tf.shape(y_data)[2], dtype=tf.float32)
    
    N = y_model.shape[-1]


    mmd_12 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_model - tf.repeat(tf.transpose(y_data, perm=(2,1,0)), repeats=N, axis=2)), axis=1)+K.epsilon()))
    mmd_22 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_model - tf.repeat(tf.transpose(y_model, perm=(2,1,0)), repeats=N, axis=2)), axis=1)+K.epsilon()))

    loss = 2*mmd_12/(n_samples_model*n_samples_data) -  mmd_22/(n_samples_model*(n_samples_model-1))
    return loss


# subclass Keras loss
class MaxMeanDiscrepancy(Loss):
    def __init__(self, beta=1.0, name="MaxMeanDiscrepancy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta

    def call(self, y_data, y_model):
        return ed(y_data, y_model)

    def get_config(self):
        cfg = super().get_config()
        cfg['beta'] = self.beta
        return cfg



################ define soft rank layer ################
class SoftRank(layers.Layer):
    """Differentiable ranking layer"""
    def __init__(self, alpha=1000.0):
        super(SoftRank, self).__init__()
        self.alpha = alpha # constant for scaling the sigmoid to approximate sign function, larger values ensure better ranking, overflow is handled properly by tensorflow

    def call(self, inputs, training=None):
        # input is a ?xSxD tensor, we wish to rank the S samples in each dimension per each batch
        # output is  ?xSxD tensor where for each dimension the entries are (rank-0.5)/N_rank
        if training:
            x = tf.expand_dims(inputs, axis=-1) #(?,S,D) -> (?,S,D,1)
            x_2 = tf.tile(x, (1,1,1,tf.shape(x)[1])) # (?,S,D,1) -> (?,S,D,S) (samples are repeated along axis 3, i.e. the last axis)
            x_1 = tf.transpose(x_2, (0,3,2,1)) #  (?,S,D,S) -> (?,S,D,S) (samples are repeated along axis 1)
            return tf.transpose(tf.reduce_sum(tf.sigmoid(self.alpha*(x_1-x_2)), axis=1), perm=(0,2,1))/(tf.cast(tf.shape(x)[1], dtype=tf.float32))
        return inputs
    
    def get_config(self):
        return {"alpha": self.alpha}


