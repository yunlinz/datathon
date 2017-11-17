import pymc3 as pm
import theano as th
from theano.tensor.nlinalg import diag
import matplotlib.pyplot as plt

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

    def make_func(self):
      raise NotImplementedError

    def setup_model(self, data):
        with pm.Model() as model:
            self.transmat_ = pm.Normal('Tmat', mu=0, sd=1, shape=(self.latent_dimension))
            self.hidden_states.append(
                pm.Normal('H0', mu=0, sd=1, shape=(self.sample_minibatch, self.latent_dimension), testval=np.random.randn(self.sample_minibatch, self.latent_dimension))
            )
            for i in range(1, self.num_time_steps):
              self.hidden_states.append(
                th.dot(self.hidden_states[-1], diag(self.transmat_))
              )
            F = pm.Normal('F', mu=0, sd=1, shape=(self.latent_dimension, self.observ_dimension), testval=np.random.randn(self.latent_dimension, self.observ_dimension))
            for i in range(self.num_time_steps):
              self.observed_states.append(
                  th.dot(self.hidden_states[i], F) + pm.Normal('X_{}'.format(i), mu=0, sd=1, shape=(self.sample_minibatch, self.observ_dimension), observed=data[i])
              )
            approx = pm.fit(n = 50000, method=pm.ADVI())
            trace = approx.sample(100)
            pm.traceplot(trace)
            plt.savefig('fig.pdf')



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