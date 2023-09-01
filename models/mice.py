### MUTUAL INFORMATION COPULA ESTIMATOR : MICE #####
### non -parametric framework for sensitive analysis, showing deep relations between variables, including copula and its density ####
## GOAL : a fast first analysis #####
from __future__ import print_function, division



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
import pandas as pd 
from models.utils import *



class MICE(object):
    def __init__(self, dim_latent=10, dim_out=2, n_samples_train=200, n_layers=2, n_neurons=100, activation="relu", alpha=1000.0, mu=0.0, sigma=1.0, sigmoid_layer=False, sigmoid_slope=1.0, optimizer="Adam"):
        #copulas estimation part
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.n_samples_train = n_samples_train
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.sigmoid_layer = sigmoid_layer
        self.sigmoid_slope = sigmoid_slope
        self.optimizer = optimizer
        self.model = self._build_model()
        #density estimation part  
        self.copulas_density = self.build_copulas_density() 
        v = Input(shape=(self.dim_out,))
        v_random = Input(shape=(self.dim_out,))
        d_v = self.copulas_density(v)
        d_v_random = self.copulas_density(v_random)
        self.combined = Model([v, v_random], [d_v,d_v_random])
        self.combined.compile(loss=[my_binary_crossentropy,wasserstein_loss],loss_weights=[1,1], optimizer=Adam(0.002, 0.5))
        
    def _build_model(self):
        z_in = tf.keras.Input(shape=(self.dim_latent, self.n_samples_train), name="z_in")
        
        z = layers.Permute((2,1))(z_in)
        for i in range(self.n_layers):
            z = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation=self.activation)(z)
        z = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation="linear")(z)

        
        z = SoftRank(alpha=self.alpha)(z)
        z = layers.Permute((2,1))(z)

        mdl = Model(inputs=z_in, outputs=z)
        mdl.compile(loss=MaxMeanDiscrepancy(), optimizer=self.optimizer)

        return mdl
    
    def build_copulas_density(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.dim_out))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.3))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='softplus'))
        T = Input(shape=(self.dim_out,))
        D = model(T)
        return Model(T, D)
    
    def fit(self, y_train, regen_noise=1e12, batch_size=10, epochs=10):
        """ fit the model """
       
        training_data = y_train
        self.loss_dict = {"train": []}
       


        y_train_ = np.transpose(np.reshape(training_data,(-1, batch_size, training_data.shape[1]), order="C"), axes=(0,2,1))
        
        for i in tqdm(range(epochs)):
            if i%regen_noise == 0:
                z_train = self._make_some_noise(y_train_.shape[0])

            
            y_train_, z_train_ = shuffle(y_train_, z_train)
            loss_batch=[]
            for j in range(y_train_.shape[0]):
                loss_batch.append(self.model.train_on_batch(x=z_train_[[j],:,:], y=y_train_[[j],:,:]))
            self.loss_dict["train"].append(np.mean(loss_batch))

        self._fit_marginals()
        
        
        valid = np.ones((batch_size, 1))
      
        for epoch in (range(5000)):

            # ---------------------
            #  Train CODINE
            # ---------------------
            data_v = self.simulate(batch_size)
            
            #--------------------------c'est ici le "seul" élément qui nous importe réellement-------------------#
            data_v_random = np.random.uniform(0,1,(batch_size, self.dim_out)) #un bruit blanc aléatoire total 
            
            
            D_value_1 = self.copulas_density.predict(data_v) #T(u) lorsque u suit la loi de la copule c(u)!!
            D_value_2 = self.copulas_density.predict(data_v_random)#T(u) lorsque u suit une distribution normale multivariée unif

            d_loss = self.combined.train_on_batch([data_v,data_v_random],[valid,valid]) #prend en input les deux bruits, calcul
            
          
           

            copula_estimate = D_value_1 #T(u) 
            self_consistency = D_value_2 #la moyenne "self consistency", doit donner 1 si on a bien la densité du copule !!, premier test important à "relever"
            # Plot the progress, progression du processus 
            print ("%d [D total loss : %f, Copula estimates: %f, Self-consistency mean test: %f]" % (epoch, d_loss[0], np.mean(copula_estimate), np.mean(self_consistency)))
            
        #return pd.DataFrame(self.loss_dict)
    
    
    def _eval(self, x, y):       
        return mmd_score(x,y, drop_xx=True)


    def _fit_marginals(self, n_samples=int(1e6)):
        """ fit marginal distributions via ECDF on large set of samples """
        samples_latent = self._get_latent_samples(n_samples)
        samples_latent = np.concatenate((samples_latent,
                                        np.ones((1, samples_latent.shape[1])),
                                        np.zeros((1, samples_latent.shape[1]))), axis=0)
        self.cdfs = []
        for i in range(samples_latent.shape[1]):
            self.cdfs.append(fit_ecdf(samples_latent[:,i]))
    

    def _get_latent_samples(self, n_samples):
        """ Draw samples from the latent distribution """
        return np.reshape(np.transpose(self.model.predict(self._make_some_noise(np.ceil(n_samples/self.n_samples_train).astype(int))), (0,2,1)), (-1,self.dim_out))[0:n_samples,:]


    def simulate(self, n_samples=100, return_latent_samples=False):
        """ draw n_samples randomly from distribution  """
        samples_latent = self._get_latent_samples(n_samples)
        samples = []
        for i in range(self.dim_out):
            samples.append(self.cdfs[i](samples_latent[:,i]))
        if return_latent_samples:
            return np.column_stack(samples), samples_latent
        else:
            return np.column_stack(samples)


    def cdf(self, v, u=None, n=10000):
        """ evaluate the empirical copula at points u using n samples"""
        # cdf is evaluated at points v, v has to be a MxD vector in [0,1]^D, cdf is evaluated at these points
        # u are samples from model NxD vector in [0,1]^D, if None n points will be sampled
        # larger n will lead to better estimation of the empirical copula but slows down computation

        if u is None:
            u = self.simulate(n)
        cdf_vals = np.empty(shape=(len(v)))
        for i in range(v.shape[0]):
            cdf_vals[i] = np.sum(np.all(u<=v[[i],:], axis=1))
        return cdf_vals/len(u)
    
    
    def get_cdf(self, u=None, n=10000, grid=np.arange(0.0, 1.1, 0.1)):
        """ Obtain a linearly interpolated cdf on specified grid. Very slow for large n or fine grid in d>3 """
        if u is None:
            u = self.simulate(n) # draw samples
        mgrid = np.meshgrid(*[grid for i in range(self.dim_out)], indexing="ij") # prepare grid in d dimensions
        mgrid = np.column_stack([np.ravel(i) for i in mgrid]) # reshape
        c_vals = np.empty(len(mgrid))
        for i in range(len(mgrid)):
            c_vals[i] = np.sum(np.all(u<=mgrid[[i],:], axis=1)) # compute empirical cdf for each grid point
        C_rs = np.reshape(c_vals/len(u),[len(grid) for i in range(self.dim_out)], order="C")
        cdf_fun = RegularGridInterpolator([grid for i in range(self.dim_out)], C_rs) # obtain linear interpolator function for grid
        return cdf_fun
    
    
    def _make_some_noise(self, n):
        """ returns normally distributed noise of dimension (N,DIM_LATENT,N_SAMPLES_TRAIN) """
        return np.random.normal(loc=self.mu, scale=self.sigma, size=(n, self.dim_latent, self.n_samples_train))
    def mutual_information(self):
        return np.mean(np.log(np.squeeze(self.copulas_density.predict(self.simulate(1000000)))))

    def get_local_mutual_index(self,alpha,beta,x):
        #real values of the variable x
        #first step about local mutual index. Important contribution of the article, from global to local sensitivity analysis
        ecdf = fit_ecdf(x) #starting by fitting the distribution 
        a,b = ecdf(alpha),ecdf(beta)
        local_mutual_index_y_database=(self.simulate(1000000)) #1 million points 
        local_density_in_these_points_y = self.copulas_density(local_mutual_index_y_database)
        points = pd.DataFrame()
        points["dense"] = np.squeeze(local_density_in_these_points_y)
        points["x"] = local_mutual_index_y_database[:,0]
        points["f"] = local_mutual_index_y_database[:,1]
        points = df[df["x"]>a]
        points = df[df["x"]<b]
        return np.mean(np.log(points["dense"]))
    
        