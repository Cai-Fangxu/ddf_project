from scipy.integrate import odeint
import numpy as np

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
