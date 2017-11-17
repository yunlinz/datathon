import edward as ed
import tensorflow as tf

from edward.models import Normal, WishartFull, MultivariateNormalFullCovariance, MultivariateNormalDiag, Empirical, WishartCholesky
from tensorflow.contrib.distributions import WishartCholesky
class AbstractSSModel(object):
    def __init__(self, n, d, k, T):
        self.function = None
        self.latent_dimension = d
        self.observation_dim = k
        self.time_steps = T

        self.seed = 0
        self.obs = []
        self.latent_vars = []
        self.parameters = []
        self.trans_mat = None

        self.qLatentVars = []
        self.qParameters = []
        self.qTransMat = None
        self.set_up_model()


    def make_func(self):
        """
        Implements a tensorflow mapping from d dimensions to k, set self.function to a function that take in tensor
        with shape matching that of latent dimension and outputs one with dimension of observation
        :return: None
        """
        raise NotImplementedError

    def set_up_model(self):
        self.trans_mat = Normal(loc=tf.zeros([self.latent_dimension, self.latent_dimension])+tf.eye(self.latent_dimension),
                                scale=tf.ones([self.latent_dimension, self.latent_dimension]))
        self.latent_vars.append(Normal(loc=tf.zeros([1,3]), scale=tf.ones([1,3]), name='H_0')) # we use 0-index
        for i in range(1, self.time_steps):
            self.latent_vars.append(
                Normal(loc = tf.zeros([1,3]), scale=tf.ones([1,3]), name = 'H_{}'.format(i))
            )

        self.make_func()
        for i in range(self.time_steps):
            self.obs.append(
                self.function(self.latent_vars[i])
            )


    def train(self, data, T=10000):
        assert len(data) == len(self.obs)

        data_dict = {x_var: x for x_var, x in zip(self.obs, data)}

        latent_var_dict = {}
        for h_var in self.latent_vars:
            self.qLatentVars.append(
                Normal(loc=tf.Variable(tf.zeros([1, self.latent_dimension])),
                       scale=tf.nn.softplus(tf.zeros([1, self.latent_dimension])))
            )
            latent_var_dict[h_var] = self.qLatentVars[-1]
            assert h_var.shape == self.qLatentVars[-1].shape, "{} =!= {}".format(h_var.shape, self.qLatentVars[-1].shape)

        self.qTransMat =  Normal(loc=tf.Variable(tf.ones([self.latent_dimension,
                                                             self.latent_dimension]) /
                                                            self.latent_dimension ** 2),
                                 scale=tf.nn.softplus(tf.Variable(tf.ones([self.latent_dimension,
                                                             self.latent_dimension]) )))
        assert self.trans_mat.shape == self.qTransMat.shape
        other_dict = {self.trans_mat: self.qTransMat}
        param_dict = self.init_params(T)

        feed_dict = {**latent_var_dict, **other_dict, **param_dict}

        inference = ed.inferences.KLpq(feed_dict, data=data_dict)
        #inference.initialize(step_size=200)
        inference.run(n_iter=1000000)
        print(self.qTransMat.sample(1000).eval().mean(axis=0))

    def init_params(self, T):
        raise NotImplementedError

class LinearSSModel(AbstractSSModel):
    def __init__(self, *args, **kwargs):
        super(LinearSSModel, self).__init__(*args, **kwargs)

    def make_func(self):
        self.parameters.append(Normal(loc=tf.zeros([self.latent_dimension, self.observation_dim]),
                                                            scale=tf.ones([self.latent_dimension, self.observation_dim])
                                                            )
                               )
        self.parameters.append(Normal(loc=tf.zeros([self.observation_dim, self.observation_dim]) + tf.eye(self.observation_dim),
                                      scale=tf.ones([self.observation_dim, self.observation_dim])))
        self.function = lambda x: Normal(loc=tf.matmul(x, self.parameters[0]),
                                        scale=tf.ones([1, self.observation_dim]),
                                       sample_shape=1000)

    def init_params(self, T):
        param_dict = {}
        self.qParameters.append(
            Normal(loc=tf.Variable(tf.zeros([self.latent_dimension, self.observation_dim])),
                   scale=tf.nn.softplus(tf.zeros([self.latent_dimension, self.observation_dim])))
        )

        param_dict[self.parameters[0]] = self.qParameters[0]
        assert self.parameters[0].shape == self.qParameters[0].shape
        self.qParameters.append(
            Normal(loc=tf.Variable(tf.ones([self.observation_dim, self.observation_dim]) /
                                                          self.observation_dim ** 2),
                   scale=tf.nn.softplus(tf.zeros([self.observation_dim, self.observation_dim])))
        )
        param_dict[self.parameters[1]] = self.qParameters[1]
        assert self.parameters[1].shape == self.qParameters[1].shape
        return param_dict


if __name__ == '__main__':
    # generate some fake data and try to do an inference
    import numpy as np
    np.random.seed(0)
    h0 = np.random.randn(1000, 3)
    T = np.eye(3)
    F = np.random.randn(3, 100)
    A_temp = np.random.rand(100,100)
    A_temp = np.dot(A_temp.T, A_temp)
    L = np.linalg.cholesky(A_temp)
    h1 = np.dot(h0, T) + np.random.randn(1000,3) / 10
    h2 = np.dot(h0, T) + np.random.randn(1000,3) / 10

    x0 = np.dot(np.dot(h0, F), L.T).reshape((1000,100)).reshape((1000,1,100))
    x1 = np.dot(np.dot(h1, F), L.T).reshape((1000,100)).reshape((1000,1,100))
    x2 = np.dot(np.dot(h2, F), L.T).reshape((1000,100)).reshape((1000,1,100))

    model = LinearSSModel(1000, 3, 100, 3)
    model.train([x0, x1, x2])