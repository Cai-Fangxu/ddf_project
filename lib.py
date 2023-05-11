import jax
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from functools import partial
from tqdm import tqdm
import optax
import sklearn

class stimulus_base_class():
    def __init__(self, time_range, initial_state=None, time_constant=10, rescale_factor=1, dc_stimulus=0) -> None:
        self.time_range = time_range # (start, end, spacing)
        self.initial_state = initial_state
        self.time_constant = time_constant
        self.rescale_factor = rescale_factor
        self.dc_stimulus = dc_stimulus
        self.sol = None
        self.stimulus_list = None

    def dfdt(self, state, t):
        pass

    def get_stimulus_list(self):
        pass

class lorenz1996_stimulus(stimulus_base_class):
    def __init__(self, N: int, F:int, time_range, initial_state, time_constant=10, rescale_factor=1, dc_stimulus=0) -> None:
        super().__init__(time_range, initial_state, time_constant, rescale_factor, dc_stimulus)
        self.N = N
        self.F = F
    
    def dfdt(self, state, t):
        x_l1 = np.roll(state, -1)
        x_r2 = np.roll(state, 2)
        x_r1 = np.roll(state, 1)
        return ((x_l1-x_r2)*x_r1 - state + self.F)/self.time_constant

    def get_stimulus_list(self):
        self.sol = self.rescale_factor*odeint(self.dfdt, self.initial_state, np.arange(*self.time_range))
        self.stimulus_list = self.sol[:, 0] + self.dc_stimulus

class lorenz1963_stimulus(stimulus_base_class):
    def __init__(self, time_range, initial_state=[-3.1, -3.1, 20.7], rho:float=28, sigma:float=10, beta:float=8/3 ,time_constant=10, rescale_factor=1, dc_stimulus=0) -> None:
        super().__init__(time_range, initial_state, time_constant, rescale_factor, dc_stimulus)
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
    
    def dfdt(self, state, t):
        x, y, z = state[0], state[1], state[2]
        dxdt = self.sigma*(y-x)/self.time_constant
        dydt = (x*(self.rho - z) - y)/self.time_constant
        dzdt = (x*y - self.beta*z)/self.time_constant
        return [dxdt, dydt, dzdt]

    def get_stimulus_list(self):
        self.sol = self.rescale_factor*odeint(self.dfdt, self.initial_state, np.arange(*self.time_range))
        self.stimulus_list = self.sol[:, 0] + self.dc_stimulus

class colpitts_stimulus(stimulus_base_class):
    def __init__(self, time_range, initial_state=[0.1, 0.1, 0.1], alpha:float=5, gamma:float=0.0797, eta:float=6.273 ,q:float=0.6898, time_constant=10, rescale_factor=1, dc_stimulus=0) -> None:
        super().__init__(time_range, initial_state, time_constant, rescale_factor, dc_stimulus)
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.q = q
    
    def dfdt(self, state, t):
        x1, x2, x3 = state[0], state[1], state[2]
        dx1dt = self.alpha*x2/self.time_constant
        dx2dt = (-self.gamma*(x1+x3)-self.q*x2)/self.time_constant
        dx3dt = self.eta*(x2+1-np.exp(-x1))/self.time_constant
        return [dx1dt, dx2dt, dx3dt]

    def get_stimulus_list(self, variable_idx=0):
        self.sol = self.rescale_factor*odeint(self.dfdt, self.initial_state, np.arange(*self.time_range))
        self.stimulus_list = self.sol[:, variable_idx] + self.dc_stimulus

class sinusoidals_stimulus(stimulus_base_class):
    def __init__(self, time_range, amplitude, periods, coeffs, phases, dc_stimulus=0) -> None:
        """amplitude: the overall amplitude of the stimulus
        periods: a list of periods for sinusodial waves, the unit is ms"""
        super().__init__(time_range, initial_state=None, time_constant=1, rescale_factor=1, dc_stimulus=dc_stimulus)
        self.amplitude = amplitude
        self.periods = periods
        self.coeffs = coeffs
        self.phases = phases
        self.get_stimulus_list()

    def get_stimulus_list(self):
        times = np.arange(*self.time_range)
        sums = np.zeros(times.shape)
        for i in range(len(self.periods)):
            sums += self.coeffs[i]*np.sin(2*np.pi/self.periods[i]*times + self.phases[i])
        max_amp = np.max(np.abs(sums))
        sums = self.amplitude/max_amp*sums
        self.stimulus_list = sums + self.dc_stimulus

class constant_stimulus(stimulus_base_class):
    def __init__(self, dc_stimulus, time_range, time_constant=10) -> None:
        super().__init__(time_range, None, time_constant)
        n_points = len(np.arange(*time_range))
        self.stimulus_list = np.ones(n_points)*dc_stimulus

class custom_stimulus(stimulus_base_class):
    def __init__(self, stimulus_list, time_range, time_constant=10) -> None:
        super().__init__(time_range, None, time_constant)
        n_points = len(np.arange(*time_range))
        self.stimulus_list = stimulus_list

class HH_Neuron():
    def __init__(self) -> None:
        self.C = 1.0 # neuron capacitance
        self.gn = 120.0 # maximum Na conductance
        self.vna = 50.0 # Na reversal potential
        self.gk = 20.0 # maximum K conductance
        self.vk = -77.0 # K reversal potential
        self.gl = 0.3  # maximum leak condunctance
        self.vl = -54.4 # leak reversal potential

        # parameters used in the dynamics of the subunits of the ion channels
        self.vm, self.dvm = -40.0, 15.0
        self.tm0, self.tm1 = 0.1, 0.4
        self.vn, self.dvn = -55.0, 30.0
        self.tn0, self.tn1 = 1.0, 5.0
        self.vh, self.dvh  = -60.0, -15.0
        self.th0, self.th1 = 1.0, 7.0

        # parameters used in the dynamics of synapses
        self.v0, self.dv0 = -5.0, 5.0
        self.S1, self.S2 = 3.0/2.0, 5.0/3.0
        self.tau1, self.tau2 = 1.0, 3.0
        self.Ereve, self.Erevi = 0.0, -80.0

    def HH_intrinsic(self, V, m, h, n):
        """Output is the contribution to the current (positive direction is from outside to inside of the neuron) from the intrinsic dynamics of Hodgkin-Huxley neuron"""
        i_Na = self.gn*(m**3)*h*(self.vna - V)
        i_K = self.gk*(n**4)*(self.vk - V)
        i_L = self.gl*(self.vl - V)
        return i_Na + i_K + i_L

    def _g_fun(self, V, A, B):
        """Used in the dynamics of subunits of ion channels"""
        return 0.5*(1+np.tanh((V-A)/B))

    def _tau_fun(self, V, t0, t1, A, B):
        """Used in the dynamics of subunits of ion channels"""
        return t0+t1*(1.0 - np.tanh((V-A)/B)**2)

    def _S0(self, V):
        """Used in the dynamics of synapses"""
        return self._g_fun(V, self.v0, self.dv0)

    def dmdt(self, V, m):
        """dm/dt"""
        return (self._g_fun(V, self.vm, self.dvm) - m)/self._tau_fun(V, self.tm0, self.tm1, self.vm, self.dvm)

    def dndt(self, V, n):
        """dn/dt"""
        return (self._g_fun(V, self.vn, self.dvn) - n)/self._tau_fun(V, self.tn0, self.tn1, self.vn, self.dvn)
    
    def dhdt(self, V, h):
        """dh/dt"""
        return (self._g_fun(V, self.vh, self.dvh) - h)/self._tau_fun(V, self.th0, self.th1, self.vh, self.dvh)

    def dSedt(self, V_pre, Se):
        """dSe/dt"""
        s0 = self._S0(V_pre)
        return (s0 - Se)/self.tau1/(self.S1 - s0)

    def dSidt(self, V_pre, Si):
        """dSi/dt"""
        s0 = self._S0(V_pre)
        return (s0 - Si)/self.tau2/(self.S2 - s0)
    
class single_neuron_odeint():
    def __init__(self, stimulus: stimulus_base_class, neuron_initial_state) -> None:
        self.neuron = HH_Neuron()
        self.stimulus = stimulus
        self.time_range = self.stimulus.time_range
        self.neuron_initial_state = neuron_initial_state
        self.sol = None

    def dfdt(self, state, t):
        V, m, h, n = tuple(state)
        t_start, t_end, t_spacing = self.time_range
        t_idx = int(round((t-t_start)/t_spacing))
        t_idx = min(t_idx, int(round((t_end-t_start)/t_spacing))-1)
        dVdt = (self.neuron.HH_intrinsic(V, m, h, n) + self.stimulus.stimulus_list[t_idx])/self.neuron.C
        dmdt = self.neuron.dmdt(V, m)
        dhdt = self.neuron.dhdt(V, h)
        dndt = self.neuron.dndt(V, n)
        return [dVdt, dmdt, dhdt, dndt]

    def generate_sol(self):
        self.sol = odeint(self.dfdt, self.neuron_initial_state, np.arange(*self.time_range))

class mlp(nn.Module):
    network_struct: Sequence[int]

    def setup(self) -> None:
        self.layers = [nn.Dense(i) for i in self.network_struct]

    def __call__(self, x):
        x = jnp.atleast_2d(x)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            # x = nn.relu(x)
            x = nn.softplus(x)
        x = self.layers[-1](x)
        return x

class prediction_model(nn.Module):
    ann: nn.Module
    time_spacing: float
    
    def setup(self) -> None:
        self.membrane_c_inverse = self.param("1/C", jax.nn.initializers.uniform(2), (1, ))
        self.membrane_leak = self.param("leak", jax.nn.initializers.uniform(2), (1, ))

    def __call__(self, x, avg_current):
        """x is the time_delay coordinate.
        current is the average current of the interval between present time and the time of prediction. (usually it is simply (I(t) + I(t+h)/2)"""
        Vt = x[..., -1] # the voltage at time t.
        temp = self.ann(x)
        temp = temp.reshape(temp.shape[:-1]) # change shape from (..., 1) to (...)
        # return V + (temp + self.membrane_c_inverse*avg_current - self.membrane_leak*V)*self.time_spacing
        return Vt + (temp + avg_current)*self.time_spacing

class prediction_model_with_current(nn.Module):
    ann_with_current: nn.Module
    time_spacing: float
    
    def setup(self) -> None:
        self.membrane_c_inverse = self.param("1/C", jax.nn.initializers.uniform(2), (1, ))
        self.membrane_leak = self.param("leak", jax.nn.initializers.uniform(2), (1, ))

    def __call__(self, x, avg_current):
        """x is the time_delay coordinate.
        the last dim of x is [V(t-n*tau, ..., V(t-tau), V(t))]
        the last dim of avg_current is [avg_i(t-n*tau), ..., avg_i(t-tau), avg_i(t)]
        current is the average current of the interval between present time and the time of prediction. (usually it is simply (I(t) + I(t+h)/2)"""
        Vt = x[..., -1] # the voltage at time t.
        It = avg_current[..., -1] # the average current at time t
        temp = self.ann_with_current(jnp.concatenate((x, avg_current), axis=-1))
        temp = temp.reshape(temp.shape[:-1]) # change shape from (..., 1) to (...)
        # return V + (temp + self.membrane_c_inverse*avg_current - self.membrane_leak*V)*self.time_spacing
        return Vt + (temp + It)*self.time_spacing

class train_by_regression():
    """training by doing linear/ridge regression"""
    def __init__(self, centers, voltage_list, current_list, time_delay, time_delay_dim, time_spacing, beta, R, membrane_capacitance=1) -> None:
        self.centers = centers # (n_centers, time_delay_dim)
        self.voltage_list = voltage_list
        self.current_list = current_list
        self.time_delay = time_delay
        self.time_delay_dim = time_delay_dim
        self.time_spacing = time_spacing
        self.beta = beta
        self.R = R # R = 1/sigma^2
        self.membrane_capacitance = membrane_capacitance
        self.X = None
        self.Y = None

        self._pre_processing()

    def _pre_processing(self):
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training is the time_delay*(time_delay_dim-1)^th point.
        tmp_v = jnp.array([jnp.roll(self.voltage_list, -i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_v = tmp_v[:-1-first_usable_t_idx, :]

        def get_distances_to_centers(x):
            diff = self.centers - x
            dist = jnp.exp(-np.sum(diff**2, axis=-1)*self.R/2)
            return dist

        self.X = jax.vmap(get_distances_to_centers, in_axes=0)(tmp_v)
           
        tmp_delta_v = self.voltage_list[first_usable_t_idx+1:] - self.voltage_list[first_usable_t_idx:-1]
        tmp_i = (self.current_list + jnp.roll(self.current_list, -1))/2
        tmp_i = tmp_i[first_usable_t_idx:-1]
        self.Y = tmp_delta_v/self.time_spacing - tmp_i/self.membrane_capacitance

    def get_weights(self):
        ridge = sklearn.linear_model.Ridge(alpha=self.beta, fit_intercept=False)
        self.ridge = ridge.fit(self.X, self.Y)
        self.score = ridge.score(self.X, self.Y)
        return ridge.coef_

class train_by_regression_leak(train_by_regression):
    """training by doing linear/ridge regression, with leak terms built in"""

    def _pre_processing(self):
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training is the time_delay*(time_delay_dim-1)^th point.
        tmp_v = jnp.array([jnp.roll(self.voltage_list, -i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_v = tmp_v[:-1-first_usable_t_idx, :]

        def get_distances_to_centers(x):
            diff = self.centers - x
            dist = jnp.exp(-np.sum(diff**2, axis=-1)*self.R/2)
            dist = jnp.concatenate((dist, jnp.array([1., x[-1]])))
            return dist

        self.X = jax.vmap(get_distances_to_centers, in_axes=0)(tmp_v)
           
        tmp_delta_v = self.voltage_list[first_usable_t_idx+1:] - self.voltage_list[first_usable_t_idx:-1]
        tmp_i = (self.current_list + jnp.roll(self.current_list, -1))/2
        tmp_i = tmp_i[first_usable_t_idx:-1]
        self.Y = tmp_delta_v/self.time_spacing - tmp_i/self.membrane_capacitance


class train():
    def __init__(self, 
        stimulus_list: Sequence[float], # 1d array
        voltage_list: Sequence[float], # 1d array
        model: prediction_model,
        time_delay: int,
        time_delay_dim: int,
        time_range: Sequence[int], #(start, end, spacing)
        batch_size: int):

        self.stimulus_list = stimulus_list
        self.voltage_list = voltage_list
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.time_range = time_range
        self.batch_size = batch_size
        self.n_batches = None
        self.batches = None 
        # the shape of batches is (n_batches, batch_size, time_delay_dim + 2)
        # the last dimension of batches is (V(t-n*tau), V(t-(n-1)*tau), ..., V(t-tau), V(t))
        self.loss_list = None

    def get_batches(self, key):
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training is the time_delay*(time_delay_dim-1)^th point.
        n_points = len(self.voltage_list) - first_usable_t_idx - 1 # -1 because the prediction always needs the data at the next time step
        tmp_v = jnp.array([jnp.roll(self.voltage_list, -i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_v = tmp_v[:-1-first_usable_t_idx, :]
        tmp_pred_v = self.voltage_list[first_usable_t_idx+1:].reshape((-1, 1))
        tmp_i = (self.stimulus_list + jnp.roll(self.stimulus_list, -1))/2
        tmp_i = tmp_i[first_usable_t_idx:-1].reshape((-1, 1))
        tmp = jnp.concatenate((tmp_v, tmp_i, tmp_pred_v), axis=-1) # dim of tmp is (n_points, time_delay_dim+2)
        tmp = jax.random.permutation(key, tmp, axis=0, independent=False)
        self.n_batches = int(n_points/self.batch_size)
        self.batches = tmp[:self.n_batches*self.batch_size, :].reshape((self.n_batches, self.batch_size, -1))
        
    @partial(jax.jit, static_argnums=(0, ))
    def loss(self, params, batch):
        xs = batch[:, :-2]
        stimulus_current = batch[:, -2]
        preds_true = batch[:, -1]
        preds = self.model.apply(params, xs, stimulus_current)
        return jnp.sum((preds - preds_true)**2)/self.batch_size

    @partial(jax.jit, static_argnums=(0, ))
    def loss_and_grad(self, params, batch):
        return jax.value_and_grad(self.loss, argnums=0)(params, batch)

    def run(self, params, n_epochs, optimizer):
        pbar = tqdm(range(n_epochs))
        tx = optimizer
        # tx = optax.sgd(learning_rate=lr, momentum=0.9)
        opt_state = tx.init(params)
        self.loss_list = np.zeros(n_epochs)
        for i in pbar:
            batch_idx = i%self.n_batches
            # print(batch_idx)
            loss_val, grads = self.loss_and_grad(params, self.batches[batch_idx])
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.loss_list[i] = loss_val
            pbar.set_postfix({"loss": loss_val})
        return params

class train_with_current():
    def __init__(self, 
        stimulus_list: Sequence[float], # 1d array
        voltage_list: Sequence[float], # 1d array
        model: prediction_model_with_current,
        time_delay: int,
        time_delay_dim: int,
        time_range: Sequence[int], #(start, end, spacing)
        batch_size: int):

        self.stimulus_list = stimulus_list
        self.voltage_list = voltage_list
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.time_range = time_range
        self.batch_size = batch_size
        self.n_batches = None
        self.batches = None 
        # the shape of batches is (n_batches, batch_size, time_delay_dim + 2)
        # the last dimension of batches is (V(t-n*tau), V(t-(n-1)*tau), ..., V(t-tau), V(t))
        self.loss_list = None

    def get_batches(self, key):
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training is the time_delay*(time_delay_dim-1)^th point.
        n_points = len(self.voltage_list) - first_usable_t_idx - 1 # -1 because the prediction always needs the data at the next time step
        tmp_v = jnp.array([jnp.roll(self.voltage_list, -i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_v = tmp_v[:-1-first_usable_t_idx, :]

        tmp_i = ((self.stimulus_list + jnp.roll(self.stimulus_list, -1))/2)[:-1]
        tmp_i = jnp.array([jnp.roll(tmp_i, -i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_i = tmp_i[:-first_usable_t_idx, :]

        tmp_pred_v = self.voltage_list[first_usable_t_idx+1:].reshape((-1, 1))

        tmp = jnp.concatenate((tmp_v, tmp_i, tmp_pred_v), axis=-1) # dim of tmp is (n_points, time_delay_dim+2)
        tmp = jax.random.permutation(key, tmp, axis=0, independent=False)
        self.n_batches = max(int(round(n_points/self.batch_size))-1, 1)
        self.batches = tmp[:self.n_batches*self.batch_size, :].reshape((self.n_batches, self.batch_size, 2*self.time_delay_dim+1))
        
    @partial(jax.jit, static_argnums=(0, ))
    def loss(self, params, batch):
        xs = batch[:, :self.time_delay_dim]
        stimulus_current = batch[:, self.time_delay_dim:2*self.time_delay_dim]
        preds_true = batch[:, -1]
        preds = self.model.apply(params, xs, stimulus_current)
        return jnp.sum((preds - preds_true)**2)/self.batch_size

    @partial(jax.jit, static_argnums=(0, ))
    def loss_and_grad(self, params, batch):
        return jax.value_and_grad(self.loss, argnums=0)(params, batch)

    def run(self, params, n_epochs, optimizer):
        pbar = tqdm(range(n_epochs))
        tx = optimizer
        # tx = optax.sgd(learning_rate=lr, momentum=0.9)
        opt_state = tx.init(params)
        self.loss_list = np.zeros(n_epochs)
        for i in pbar:
            batch_idx = i%self.n_batches
            # print(batch_idx)
            loss_val, grads = self.loss_and_grad(params, self.batches[batch_idx])
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.loss_list[i] = loss_val
            pbar.set_postfix({"loss": loss_val})
        return params

class train_with_current_multi_prediction():
    def __init__(self, 
        stimulus_list: Sequence[float], # 1d array
        voltage_list: Sequence[float], # 1d array
        model: prediction_model_with_current,
        time_delay: int,
        time_delay_dim: int,
        time_range: Sequence[int], #(start, end, spacing)
        batch_size: int, 
        n_prediction_steps: int):

        self.stimulus_list = stimulus_list
        self.voltage_list = voltage_list
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.time_range = time_range
        self.batch_size = batch_size
        self.n_batches = None
        self.batches = None 
        # the shape of batches is (n_batches, batch_size, 2((time_delay_dim-1)*time_delay + 1 + n_prediction_steps)-1)
        # the last dimension of batches is (V(t-n*tau), V(t-(n-1)*tau), ..., V(t+m*tau), avg_i(t-n*tau), avg_i(t-(n-1)*tau), ..., avg_i(t+m*tau))
        self.n_prediction_steps = n_prediction_steps
        self.loss_list = None

    def get_batches(self, key):
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training is the time_delay*(time_delay_dim-1)^th point.
        n_points = len(self.voltage_list) - first_usable_t_idx - self.n_prediction_steps 

        tmp_v = jnp.array([self.voltage_list[t: t+first_usable_t_idx+self.n_prediction_steps+1] for t in range(n_points)])

        tmp_i = ((self.stimulus_list + jnp.roll(self.stimulus_list, -1))/2)[:-1]
        tmp_i = jnp.array([tmp_i[t: t+first_usable_t_idx+self.n_prediction_steps] for t in range(n_points)]) # note that tmp_i is one element shorter than tmp_v in the last dim. 

        tmp = jnp.concatenate((tmp_v, tmp_i), axis=-1) # dim of tmp is (n_points, time_delay_dim+2)
        tmp = jax.random.permutation(key, tmp, axis=0, independent=False)
        self.n_batches = max(int(round(n_points/self.batch_size))-1, 1)
        self.batches = tmp[:self.n_batches*self.batch_size, :].reshape((self.n_batches, self.batch_size, 2*((self.time_delay_dim-1)*self.time_delay + 1 + self.n_prediction_steps)-1))
        
    @partial(jax.jit, static_argnums=(0, ))
    def loss(self, params, batch):
        voltages = batch[:, :(self.time_delay_dim-1)*self.time_delay + 1]
        stimulus_current = batch[:, (self.time_delay_dim-1)*self.time_delay + 1 + self.n_prediction_steps:]
        preds_true = batch[:, (self.time_delay_dim-1)*self.time_delay + 1 : (self.time_delay_dim-1)*self.time_delay + 1 + self.n_prediction_steps]
        state_indices = jnp.array([n*self.time_delay for n in range(self.time_delay_dim)])
        for i in range(self.n_prediction_steps):
            vs = voltages[:, state_indices]
            avg_is = stimulus_current[:, state_indices]
            next_v = self.model.apply(params, vs, avg_is).reshape((-1, 1))
            voltages = jnp.concatenate((voltages, next_v), axis=-1)
            state_indices = state_indices + 1
    
        preds = voltages[:, -self.n_prediction_steps: ]
        return jnp.sum((preds - preds_true)**2)/self.batch_size

    @partial(jax.jit, static_argnums=(0, ))
    def loss_and_grad(self, params, batch):
        return jax.value_and_grad(self.loss, argnums=0)(params, batch)

    def run(self, params, n_epochs, optimizer):
        pbar = tqdm(range(n_epochs))
        tx = optimizer
        # tx = optax.sgd(learning_rate=lr, momentum=0.9)
        opt_state = tx.init(params)
        self.loss_list = np.zeros(n_epochs)
        for i in pbar:
            batch_idx = i%self.n_batches
            # print(batch_idx)
            loss_val, grads = self.loss_and_grad(params, self.batches[batch_idx])
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.loss_list[i] = loss_val
            pbar.set_postfix({"loss": loss_val})
        return params

# don't use!
class train_multistep_prediction():
    def __init__(self, 
        stimulus_list: Sequence[float], # 1d array
        voltage_list: Sequence[float], # 1d array
        model: prediction_model,
        time_delay: int,
        time_delay_dim: int,
        time_range: Sequence[int], #(start, end, spacing)
        batch_size: int, 
        n_prediction_steps: int=1):

        self.stimulus_list = stimulus_list
        self.voltage_list = voltage_list
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.time_range = time_range
        self.batch_size = batch_size
        self.n_prediction_steps = n_prediction_steps
        self.n_batches = None
        self.batches = None 
        # the shape of batches is (n_batches, batch_size, 2*time_delay_dim + 1)
        # the last dimension of batches is (V(t), V(t-\tau), V(t-2\tau), V(t-3\tau), ...)
        self.loss_list = None

    def get_batches(self, key):
        t_start, t_end, t_spacing = self.time_range
        first_usable_t_idx = self.time_delay*(self.time_delay_dim-1) # the first point that can be used for training is the time_delay*(time_delay_dim-1)^th point.
        n_points = int((t_end-t_start)/t_spacing) - first_usable_t_idx - self.n_prediction_steps # -n_prediction_steps because the prediction always needs the data at the next time step
        tmp_v = jnp.array([jnp.roll(self.voltage_list, i*self.time_delay) for i in range(self.time_delay_dim)]).T
        tmp_v = tmp_v[first_usable_t_idx:-self.n_prediction_steps, :]

        tmp_pred_v = jnp.array([jnp.roll(self.voltage_list[first_usable_t_idx+1:], -i) for i in range(self.n_prediction_steps)]).T
        tmp_pred_v = tmp_pred_v[:len(tmp_pred_v)+1-self.n_prediction_steps]
        tmp_pred_v = tmp_pred_v.reshape((-1, self.n_prediction_steps))

        tmp_i = (self.stimulus_list + jnp.roll(self.stimulus_list, -1))/2
        tmp_i = tmp_i[first_usable_t_idx:-1]
        tmp_i = jnp.array([jnp.roll(tmp_i, -i) for i in range(self.n_prediction_steps)]).T
        tmp_i = tmp_i[:len(tmp_i)+1-self.n_prediction_steps]
        tmp_i = tmp_i.reshape((-1, self.n_prediction_steps))

        tmp = jnp.concatenate((tmp_v, tmp_i, tmp_pred_v), axis=-1) # dim of tmp is (n_points, time_delay_dim+2)
        tmp = jax.random.permutation(key, tmp, axis=0, independent=False)
        self.n_batches = int(n_points/self.batch_size)
        self.batches = tmp[:self.n_batches*self.batch_size, :].reshape((self.n_batches, self.batch_size, -1))
        
    @partial(jax.jit, static_argnums=(0, ))
    def loss(self, params, batch):
        xs = batch[:, :-2*self.n_prediction_steps]
        avg_Is = batch[:, -2*self.n_prediction_steps:-self.n_prediction_steps]
        preds_true = batch[:, -self.n_prediction_steps:]
        for i in range(self.n_prediction_steps):
            preds_tmp = self.model.apply(params, xs[:, i:self.time_delay_dim+i], avg_Is[:, i])
            xs = jnp.concatenate((xs, preds_tmp.reshape((-1, 1))), axis=-1)
        preds = xs[:, -self.n_prediction_steps:]
        return jnp.sum((preds - preds_true)**2)/self.batch_size

    @partial(jax.jit, static_argnums=(0, ))
    def loss_and_grad(self, params, batch):
        return jax.value_and_grad(self.loss, argnums=0)(params, batch)

    def run(self, params, n_epochs, optimizer):
        pbar = tqdm(range(n_epochs))
        tx = optimizer
        # tx = optax.sgd(learning_rate=lr, momentum=0.9)
        opt_state = tx.init(params)
        self.loss_list = np.zeros(n_epochs)
        for i in pbar:
            batch_idx = i%self.n_batches
            # print(batch_idx)
            loss_val, grads = self.loss_and_grad(params, self.batches[batch_idx])
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            self.loss_list[i] = loss_val
            pbar.set_postfix({"loss": loss_val})
        return params

class generate_prediction():
    def __init__(self, model: prediction_model, time_delay:int, time_delay_dim: int, stimulus_list: Sequence[float], initial_piece: Sequence[float]) -> None:
        """the fist element of stimulus_list should be the stimulus current at t0.
        the initial_piece should be a sequence of voltages from (t0 - time_delay*tine_delay_dim) to t0. (the length of this sequence is time_delay*(time_delay_dim-1)+1)
        """
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.avg_i = ((stimulus_list + jnp.roll(stimulus_list, -1))/2)[:-1]
        self.initial_piece = initial_piece
        self.prediction_list = np.concatenate((self.initial_piece, np.zeros(len(self.avg_i))))

    def run(self, params, n_steps=None):
        initial_piece_len = self.time_delay*(self.time_delay_dim-1)+1
        v_indices = np.array([n*self.time_delay for n in range(self.time_delay_dim)]) 
        if n_steps is not None:
            n_steps = min(n_steps, len(self.avg_i))
        for i in tqdm(range(n_steps)):
            state = jnp.array(self.prediction_list[v_indices])
            next_v = self.model.apply(params, state, self.avg_i[i])
            self.prediction_list[i+initial_piece_len] = next_v
            v_indices = v_indices + 1

class generate_prediction_with_current():
    def __init__(self, model: prediction_model_with_current, time_delay:int, time_delay_dim: int, stimulus_list: Sequence[float], initial_piece: Sequence[float]) -> None:
        """
        the initial_piece should be a sequence of voltages from (t0 - time_delay*(time_delay_dim-1)) to t0. (the length of this sequence is time_delay*(time_delay_dim-1)+1)
        the stimulus list should also starts at (t0 - time_delay*(time_delay_dim-1))
        """
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.avg_i = ((stimulus_list + jnp.roll(stimulus_list, -1))/2)[:-1]
        self.initial_piece = initial_piece
        self.prediction_list = np.zeros(len(self.avg_i)+1)
        self.prediction_list[:len(initial_piece)] = initial_piece

    def run(self, params, n_steps=None):
        # initial_piece_len = self.time_delay*(self.time_delay_dim-1)+1
        initial_piece_len = len(self.initial_piece)
        state_indices = np.array([n*self.time_delay for n in range(self.time_delay_dim)]) 
        if n_steps is not None:
            n_steps = min(n_steps, len(self.prediction_list)-initial_piece_len)
        for i in tqdm(range(n_steps)):
            vs = jnp.array(self.prediction_list[state_indices])
            avg_is = jnp.array(self.avg_i[state_indices])
            next_v = self.model.apply(params, vs, avg_is)
            self.prediction_list[i+initial_piece_len] = next_v
            state_indices = state_indices + 1

class generate_batch_prediction_with_current():
    def __init__(self, model: prediction_model_with_current, time_delay:int, time_delay_dim: int, batch_stimulus_list: Sequence[float], batch_initial_piece: Sequence[float]) -> None:
        """
        the initial_piece should be a sequence of voltages from (t0 - time_delay*(time_delay_dim-1)) to t0. (the length of this sequence is time_delay*(time_delay_dim-1)+1)
        the stimulus list should also starts at (t0 - time_delay*(time_delay_dim-1))
        """
        self.model = model
        self.time_delay, self.time_delay_dim = time_delay, time_delay_dim
        self.batch_avg_i = ((batch_stimulus_list + jnp.roll(batch_stimulus_list, -1, axis=-1))/2)[..., :-1]
        self.batch_initial_piece = batch_initial_piece
        prediction_list_shape = list(self.batch_avg_i.shape)
        prediction_list_shape[-1] += 1
        self.prediction_list = np.zeros(prediction_list_shape)
        self.prediction_list[..., :batch_initial_piece.shape[-1]] = batch_initial_piece

    def run(self, params, n_steps=None):
        # initial_piece_len = self.time_delay*(self.time_delay_dim-1)+1
        initial_piece_len = self.batch_initial_piece.shape[-1]
        state_indices = np.array([n*self.time_delay for n in range(self.time_delay_dim)]) 
        if n_steps is not None:
            n_steps = min(n_steps, self.prediction_list.shape[-1] - initial_piece_len)
        for i in tqdm(range(n_steps)):
            vs = jnp.array(self.prediction_list[..., state_indices])
            avg_is = jnp.array(self.batch_avg_i[..., state_indices])
            next_v = self.model.apply(params, vs, avg_is)
            self.prediction_list[..., i+initial_piece_len] = next_v
            state_indices = state_indices + 1