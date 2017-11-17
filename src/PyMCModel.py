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

with pm.Model() as model:
    H0 = pm.Normal('H0', mu=0, sd=1, shape=(N, latent_dim), testval=np.random.randn(N, latent_dim))
    T  = pm.Normal('T', mu=1, sd=1, shape=(latent_dim))

    Tmat = diag(T)
    H1 = th.dot(H0, Tmat)

    H2 = th.dot(H1, Tmat)

    F = pm.Normal('F', mu=0, sd=1, shape=(latent_dim, obs_dim), testval=np.random.randn(latent_dim, obs_dim))
    X0 = pm.Normal('X0', mu=th.dot(H0, F), sd=1, observed=x0)
    X1 = pm.Normal('X1', mu=th.dot(H1, F), sd=1, observed=x1)
    X2 = pm.Normal('X2', mu=th.dot(H2, F), sd=1, observed=x2)

    step1 = pm.HamiltonianMC([F])
    step2 = pm.HamiltonianMC([H0, T])
    trace = pm.sample(500, [step1, step2])
    pm.traceplot(trace)
    plt.savefig('trace.pdf')