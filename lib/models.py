import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Union
    
class mlps(nn.Module):
    def setup(self) -> None:
        self.layer_y0 = nn.Dense(features=10) # preprocess time delay embedded current
        self.layer_y1 = nn.Dense(features=10)
        self.layer_y2 = nn.Dense(features=10)
        self.layer_y3 = nn.Dense(features=1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)
        self.layer_z1 = nn.Dense(features=20)
        self.layer_z2 = nn.Dense(features=1)

        self.activation_last = lambda x: nn.activation.sigmoid(x) - jnp.log(2)

    def __call__(self, RBFs, time_delay_I):
        RBFs = jnp.atleast_2d(RBFs)
        time_delay_I = jnp.atleast_2d(time_delay_I)
        return jax.vmap(self.single_evaluation, in_axes=0)(RBFs, time_delay_I)
    
    def single_evaluation(self, RBFs, time_delay_I):
        y = self.layer_y0(time_delay_I)
        y = jnp.concatenate((RBFs, y))
        y = self.layer_y1(y)
        y = nn.activation.softplus(y)
        y = self.layer_y2(y)
        y = nn.activation.softplus(y)
        yf = self.layer_y3(y)

        z = self.layer_z1(RBFs)
        z = nn.activation.softplus(z)
        zf = self.layer_z2(z)

        return yf*zf # zf takes care of the magnitude (like a gate variable), yf takes care of the detailed shape. 
    
class prediction_model(nn.Module):
    time_spacing: float
    R: Union[Sequence[float], float]
    centers: Sequence[float] # shape = (n_centers, time_delay_dim)
    weights_rbf: Sequence[float]
    ann: nn.Module = None
    weights_leak: Sequence[float] = jnp.zeros(2)
    weight_C: float = 1.
    weights_poly: Union[Sequence[float], None] = None
    
    def setup(self) -> None:
        self.C_inverse_correction_factor = self.param("correction factor of 1/C", jax.nn.initializers.ones, (1, ))

    def __call__(self, time_delay_Vs, time_delay_avg_Is):
        """`time_delay_V` and `time_delay_avg_I` should be 1d or 2d arrays. If they are 2d, their first dimension should be n_batch. 
        The last dim of `time_delay_V` is [V(t-n*tau, ..., V(t-tau), V(t))].
        The last dim of `time_delay_avg_I` is [avg_i(t-n*tau), ..., avg_i(t-tau), avg_i(t)].
        Current is the average current of the interval between present time and the time of prediction. (usually it is simply (I(t) + I(t+h)/2)"""
        time_delay_Vs = jnp.atleast_2d(time_delay_Vs)
        time_delay_avg_Is = jnp.atleast_2d(time_delay_avg_Is)
        return jax.vmap(self.single_evaluation, in_axes=(0, 0))(time_delay_Vs, time_delay_avg_Is)
    
    def single_evaluation(self, time_delay_Vs, time_delay_avg_Is):
        """`time_delay_V` and `time_delay_avg_I` are expected to be 1d arrays"""
        Vt = time_delay_Vs[-1] # the voltage at time t.
        It = time_delay_avg_Is[-1] # the average current at time t
        rbfs = self._rbfs(time_delay_Vs)

        rbf_term = self._rbf_fun(rbfs)
        ann_term = self._ann_fun(rbfs, time_delay_avg_Is)
        leak_term = self._leaky_fun(Vt)
        poly_term = self._poly_fun(time_delay_Vs)

        return Vt + self.time_spacing*self.C_inverse_correction_factor*(rbf_term + leak_term + poly_term + It/self.weight_C) + ann_term
    
    def _rbfs(self, time_delay_Vs):
        "`time_delay_V` is expected to be 1d array"
        diff = (self.centers - time_delay_Vs)**2
        diff = jnp.sum(diff, axis=-1)*self.R/2
        return jnp.exp(-diff)
    
    def _rbf_fun(self, rbfs):
        return rbfs@self.weights_rbf
    
    def _leaky_fun(self, Vt):
        # return 0.
        return self.weights_leak[0] + self.weights_leak[1]*Vt
    
    def _poly_fun(self, time_delay_V):
        return 0.
    
    def _ann_fun(self, rbfs, time_delay_avg_Is):
        ann_out = self.ann(rbfs, time_delay_avg_Is)
        ann_out = ann_out.reshape(ann_out.shape[:-1]) # change shape from (..., 1) to (...)
        return ann_out

class prediction_model_no_ann(prediction_model):
    def _ann_fun(self, rbfs, time_delay_avg_I):
        return 0.
    
