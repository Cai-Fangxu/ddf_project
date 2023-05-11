from scipy.integrate import odeint
import numpy as np
from typing import Sequence

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
    def __init__(self, time_range, stimulus_list: Sequence[float], neuron_initial_state) -> None:
        """time_range: a tuple (start, end, spacing)
        stimulus_list: an array of I(t)
        neuron_initial_state: [V, m, h, n] at time 0"""
        self.neuron = HH_Neuron()
        self.stimulus_list = stimulus_list
        self.time_range = time_range
        self.neuron_initial_state = neuron_initial_state
        self.sol = None

    def dfdt(self, state, t):
        V, m, h, n = tuple(state)
        t_start, t_end, t_spacing = self.time_range
        t_idx = int(round((t-t_start)/t_spacing))
        t_idx = min(t_idx, int(round((t_end-t_start)/t_spacing))-1)
        dVdt = (self.neuron.HH_intrinsic(V, m, h, n) + self.stimulus_list[t_idx])/self.neuron.C
        dmdt = self.neuron.dmdt(V, m)
        dhdt = self.neuron.dhdt(V, h)
        dndt = self.neuron.dndt(V, n)
        return [dVdt, dmdt, dhdt, dndt]

    def generate_sol(self):
        self.sol = odeint(self.dfdt, self.neuron_initial_state, np.arange(*self.time_range))
