import numpy as np
import jax
import jax.numpy as jnp
import sklearn
from typing import Sequence
from functools import partial
from tqdm import tqdm
import optax
from sklearn.cluster import KMeans
import sklearn.neighbors

from . import models

def get_centers_and_R(voltage_list: Sequence[float], time_delay_V: int, time_delay_dim_V: int, n_centers: int, n_neighbors: int):
    """return a tuple: (centers, R)
    the shape of centers is (n_centers, time_delay_dim_V)
    the shape of R is (n_centers, )
    """
    points_time_delay = np.array([np.roll(voltage_list, -i*time_delay_V) for i in range(time_delay_dim_V)]).T
    points_time_delay = points_time_delay[: -time_delay_V*(time_delay_dim_V-1)]
    kmeans = KMeans(n_clusters=n_centers, random_state=0, n_init="auto").fit(points_time_delay)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(kmeans.cluster_centers_)
    distances, _ = nbrs.kneighbors(kmeans.cluster_centers_)
    sigma = distances[:, -1]
    R = 1/sigma**2
    return (kmeans.cluster_centers_, R)

class train_by_regression():
    """training by doing linear/ridge regression"""
    def __init__(self, centers, voltage_list, current_list, time_delay, time_delay_dim, time_spacing, beta, R) -> None:
        self.centers = centers # (n_centers, time_delay_dim)
        self.voltage_list = voltage_list
        self.current_list = current_list
        self.time_delay = time_delay
        self.time_delay_dim = time_delay_dim
        self.time_spacing = time_spacing
        self.beta = beta
        self.R = R # R = 1/sigma^2
        self.X = None
        self.Y = None

        self._pre_processing()

    def _pre_processing(self):
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training.
        tmp_v = jnp.array([jnp.roll(self.voltage_list, -i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_v = tmp_v[:-1-first_usable_t_idx, :]
        tmp_delta_v = self.voltage_list[first_usable_t_idx+1:] - self.voltage_list[first_usable_t_idx:-1]
        tmp_i = (self.current_list + jnp.roll(self.current_list, -1))/2
        tmp_i = tmp_i[first_usable_t_idx:-1]

        self.X = jax.vmap(self._get_basis, in_axes=(0, 0))(tmp_v, tmp_delta_v/self.time_spacing)
        self.Y = tmp_i

    def _get_basis(self, time_delay_Vs, dVdt):
        diff = self.centers - time_delay_Vs
        rbfs = jnp.exp(-np.sum(diff**2, axis=-1)*self.R/2)
        leaky_terms = jnp.array([1, time_delay_Vs[-1]])
        return jnp.concatenate((-rbfs, -leaky_terms, jnp.array([dVdt])))


    def get_weights(self, solver="auto"):
        ridge = sklearn.linear_model.Ridge(alpha=self.beta, fit_intercept=False, solver=solver)
        self.ridge = ridge.fit(self.X, self.Y)
        self.score = ridge.score(self.X, self.Y)
        weight_C = ridge.coef_[-1]
        weights_leak = ridge.coef_[-3:-1]
        weights_rbf = ridge.coef_[:-3]
        return (weights_rbf, weights_leak, weight_C)
    
    def get_error_list(self):
        return self.Y - self.X @ self.ridge.coef_

class train_by_BP():
    def __init__(self, 
        stimulus_list: Sequence[float], # 1d array
        voltage_list: Sequence[float], # 1d array
        model: models.prediction_model,
        time_delay_V: int,
        time_delay_dim_V: int,
        time_delay_I: int,
        time_delay_dim_I: int,
        time_range: Sequence[int], #(start, end, spacing)
        batch_size: int):

        self.stimulus_list = stimulus_list
        self.voltage_list = voltage_list
        self.model = model
        self.time_delay_V, self.time_delay_dim_V = time_delay_V, time_delay_dim_V
        self.time_delay_I, self.time_delay_dim_I = time_delay_I, time_delay_dim_I
        self.time_range = time_range
        self.batch_size = batch_size
        self.n_batches = None
        self.batches = None 
        # the shape of `batches` is (n_batches, batch_size, time_delay_dim_V + time_delay_dim_I + 1)
        # the last dimension of `batches` is ( V(t-xxx),...,V(t), I(t-xxx),...,I(t), V(t+1) )
        self.loss_list = None

    def get_batches(self, key):
        first_usable_t_idx_V = self.time_delay_V*(self.time_delay_dim_V-1) 
        first_usable_t_idx_I = self.time_delay_I*(self.time_delay_dim_I-1)
        first_usable_t_idx = max(first_usable_t_idx_V, first_usable_t_idx_I) # the first point that can be used for training.
        n_points = len(self.voltage_list) - first_usable_t_idx - 1 # -1 because the prediction always needs the data at the next time step
        tmp_v = jnp.array([jnp.roll(self.voltage_list, -i*self.time_delay_V) for i in range(self.time_delay_dim_V)]).T
        tmp_v = tmp_v[first_usable_t_idx-first_usable_t_idx_V : -1-first_usable_t_idx_V, :]

        tmp_i = ((self.stimulus_list + jnp.roll(self.stimulus_list, -1))/2)[:-1]
        tmp_i = jnp.array([jnp.roll(tmp_i, -i*self.time_delay_I) for i in range(self.time_delay_dim_I)]).T
        tmp_i = tmp_i[first_usable_t_idx-first_usable_t_idx_I : -first_usable_t_idx_I, :]

        tmp_pred_v = self.voltage_list[first_usable_t_idx+1:].reshape((-1, 1))

        tmp = jnp.concatenate((tmp_v, tmp_i, tmp_pred_v), axis=-1) # dim of tmp is (n_points, time_delay_dim_V + time_delay_dim_I + 1)
        tmp = jax.random.permutation(key, tmp, axis=0, independent=False)
        self.n_batches = max(int(n_points/self.batch_size)-1, 1)
        self.batches = tmp[:self.n_batches*self.batch_size, :].reshape((self.n_batches, self.batch_size, self.time_delay_dim_V+self.time_delay_dim_I+1))
        
    @partial(jax.jit, static_argnums=(0, ))
    def loss(self, params, batch):
        time_delay_V = batch[:, :self.time_delay_dim_V]
        time_dealy_I = batch[:, self.time_delay_dim_V : self.time_delay_dim_V+self.time_delay_dim_I]
        preds_true = batch[:, -1]
        preds = self.model.apply(params, time_delay_V, time_dealy_I).reshape(preds_true.shape)
        return jnp.sum((preds - preds_true)**2)/self.batch_size

    @partial(jax.jit, static_argnums=(0, ))
    def loss_and_grad(self, params, batch):
        return jax.value_and_grad(self.loss, argnums=0)(params, batch)

    def run(self, params, n_epochs, optimizer):
        pbar = tqdm(range(n_epochs))
        tx = optimizer
        opt_state = tx.init(params)
        self.loss_list = np.zeros(n_epochs)
        for i in pbar:
            batch_idx = i%self.n_batches
            loss_val, grads = self.loss_and_grad(params, self.batches[batch_idx])
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.loss_list[i] = loss_val
            pbar.set_postfix({"loss": loss_val})
        return params
    
