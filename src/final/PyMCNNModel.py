import pymc3 as pm
import theano as th
import numpy as np
from theano.tensor.nlinalg import diag
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
class PyMC3Model(object):
    def __init__(self, N, d, k, T):
      self.sample_minibatch = N
      self.latent_dimension = d
      self.observ_dimension = k
      self.num_time_steps = T
      self.hidden_states = []
      self.observed_states = []

      self.function = None
      self.model = None

    def setup_model(self, data):
#        p = 0.8
        with pm.Model() as model:
            init_states = np.random.randn(self.sample_minibatch, self.latent_dimension)
            self.hidden_states.append(
              pm.Normal('H0', mu=0, sd=1,shape=(self.sample_minibatch, self.latent_dimension), testval=init_states)

            )
            for i in range(1, self.num_time_steps):
              self.hidden_states.append(
                pm.Normal('H{}'.format(i+1), mu=self.hidden_states[-1], sd=0.1,shape=(self.sample_minibatch, self.latent_dimension), testval=init_states)
              )
	    
            l1_size = int((self.observ_dimension - self.latent_dimension)/3) + self.latent_dimension
            l2_size = int((self.observ_dimension - self.latent_dimension)/3) * 2 + self.latent_dimension  
#            P0 = pm.Bernoulli('P0', p, shape=(self.latent_dimension, l1_size), testval=np.random.binomial(1, p, size=(self.latent_dimension, l1_size)))
            W0 = pm.Normal('W0',mu=0, sd=1, shape=(self.latent_dimension, l1_size), testval=np.random.randn(self.latent_dimension, l1_size))
#            P1 = pm.Bernoulli('P1', p, shape=(l1_size, l2_size), testval=np.random.binomial(1, p, size=(l1_size, l2_size)))
            W1 = pm.Normal('W1',mu=0, sd=1, shape=(l1_size, l2_size), testval=np.random.randn(l1_size, l2_size))
            W2 = pm.Normal('W2',mu=0, sd=1, shape=(l2_size, self.observ_dimension), testval=np.random.randn(l2_size, self.observ_dimension))
	    
            for i in range(self.num_time_steps):		
                  pm.Normal('X_{}'.format(i), mu=th.dot(th.tensor.tanh(th.dot(th.tensor.tanh(th.dot(self.hidden_states[i], W0)), W1)), W2), sd=1, shape=(self.sample_minibatch, self.observ_dimension), observed=data[i])
            inference = pm.ADVI()
            iters = 150000
            approx = pm.fit(n=iters, method=inference)
            trace = approx.sample(500)

            plt.semilogy(list(range(iters)), inference.hist)
            #plt.yscale('log')i
            plt.legend()
            plt.ylabel('ELBO')
            plt.xlabel('iteration')
            plt.savefig('nn_elbo.pdf')
            import pickle
            with open('nn5d_2layer_all.pkl', 'wb') as buff:
                pickle.dump(trace, buff)

if __name__ == '__main__':
    data = np.load('../data/all_finances_all.npy')
    print(data.shape)
    T, N, k = data.shape
    model = PyMC3Model(N, 5, k, T)

    data_list = []
    for i in range(data.shape[0]):
        data_list.append(pd.DataFrame(data[i, :,:]))
    model.setup_model(data_list)

"""

latent_dim = 3
obs_dim = 100
N = 1000

import numpy as np

np.random.seed(0)
h0 = np.random.randn(N, latent_dim)
T = np.eye(latent_dim)
F = np.random.randn(latent_dim, obs_dim)
A_temp = np.random.rand(obs_dim, obs_dim)
A_temp = np.dot(A_temp.T, A_temp)
L = np.linalg.cholesky(A_temp)
h1 = np.dot(h0, T) + np.random.randn(N, latent_dim) / 10
h2 = np.dot(h0, T) + np.random.randn(N, latent_dim) / 10

x0 = np.dot(np.dot(h0, F), L.T).reshape((N, obs_dim)).reshape((N, obs_dim))
x1 = np.dot(np.dot(h1, F), L.T).reshape((N, obs_dim)).reshape((N, obs_dim))
x2 = np.dot(np.dot(h2, F), L.T).reshape((N, obs_dim)).reshape((N, obs_dim))

model = PyMC3Model(N, latent_dim, obs_dim, 3)
model.setup_model([x0, x1, x2])
"""
