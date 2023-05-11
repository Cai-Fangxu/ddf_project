import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Sequence

from . import models

class time_series_prediction():
    def __init__(self, model: models.prediction_model, time_delay_V: int, time_delay_dim_V: int, time_delay_I: int, time_delay_dim_I: int, stimulus_list: Sequence[float], initial_piece_V: Sequence[float]) -> None:
        """
        `stimulus_list` and `initial_piece_V` can be 1d or 2d (for batching) arrays. The last dimension should be the time dimension. 
        `stimulus_list` and `initial_piece_V` don't need to have the same dimension, but make sure that they are aligned in time, i.e., `stimulus_list[..., 0]` and `initial_piece_V[..., 0]` are the current and voltage at time 0 respectively. 
        The result of prediction is stored in `prediction_list`, it will be aligned with the `stimulus_list[..., 0]`. 
        `t0` is the start of the prediction. `prediction_list[..., t0+1]` will store the result of the first prediction. `t0` is automatically calculated. 
        The length of `initial_piece_V` should be `t0+1`. If it is longer, the extra part will be automatically dropped. 
        """
        self.model = model

        self.time_delay_V, self.time_delay_dim_V = time_delay_V, time_delay_dim_V
        self.time_delay_I, self.time_delay_dim_I = time_delay_I, time_delay_dim_I
        self.t0 = max(time_delay_V*(time_delay_dim_V-1), time_delay_I*(time_delay_dim_I-1))
        # the index corresponding to the start of the prediction. self.prediction_list[sellf.t0+1] (defined below) should store the first predicted voltage. 

        initial_piece_V = jnp.atleast_2d(initial_piece_V)[..., :self.t0+1] # t0 + 1 should be the length of the initial piece.
        stimulus_list = jnp.atleast_2d(stimulus_list)
        self.batch_avg_i = ((stimulus_list + jnp.roll(stimulus_list, -1, axis=-1))/2)[..., :-1]

        stimulus_len = stimulus_list.shape[-1]
        self.prediction_list = np.zeros(stimulus_list.shape)
        self.prediction_list[..., :self.t0+1] = initial_piece_V

    def run(self, params, n_steps):
        state_indices_V = np.array([self.t0 - (self.time_delay_dim_V-1-n)*self.time_delay_V for n in range(self.time_delay_dim_V)]) 
        state_indices_I = np.array([self.t0 - (self.time_delay_dim_I-1-n)*self.time_delay_I for n in range(self.time_delay_dim_I)]) 
        n_steps = min(n_steps, self.prediction_list.shape[-1] - self.t0 - 1)
        for i in tqdm(range(n_steps)):
            vs = jnp.array(self.prediction_list[..., state_indices_V])
            avg_is = jnp.array(self.batch_avg_i[..., state_indices_I])
            next_v = self.model.apply(params, vs, avg_is)
            self.prediction_list[..., i + self.t0 + 1] = next_v
            state_indices_V = state_indices_V + 1
            state_indices_I = state_indices_I + 1