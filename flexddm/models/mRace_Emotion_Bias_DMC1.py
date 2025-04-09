import numpy as np
import numba as nb
import math
from .Model import Model
from flexddm import _utilities as util

"""
Class to simulate data according to the Diffusion Model for Conflict (DMC)
with race, emotion, and distractor bias.
"""

class mRace_Emotion_Bias_DMC1(Model):

    global bounds
    global data
    global parameter_names
    global param_number

    DT = 0.01  # Time step
    VAR = 0.01  # Variance
    NTRIALS = 800  # Number of trials
    NOISESEED = 50  # Seed for noise

    def __init__(self, data=None, input_data_id="PPT", input_data_congruency="Condition", 
                 input_data_rt="RT", input_data_accuracy="Correct"):
        """
        Initializes the DMC model object.
        """
        self.modelsimulationfunction = mRace_Emotion_Bias_DMC1.model_simulation
        self.QUANTILES_CDF = [.10, .30, .50, .70, .90]
        self.QUANTILES_CAF = [.25, .50, .75]
        self.data = None

        if data is not None:
            if isinstance(data, str):
                self.data = util.getRTData(data, input_data_id, input_data_congruency, 
                                           input_data_rt, input_data_accuracy)
            else:
                self.data = data
        
        min_rt = 0.45  # Default value in case data is None
        if self.data is not None:
            min_rt = min(self.data['rt']) if not self.data.empty else 0.45  # Avoid issues if data is empty

        self.bounds = {
            "alpha": (-5, 5),
            "alpha_racial_bias": (-10, 5), 
            "beta": (0, 1),
            # "mu_c": (0.01, 0.8),
            # "shape": (1.5, 4.5),
            "characteristic_time": (0.01, 1),
            "emotion_bias": (-0.38, 0.38),
            "racial_bias": (-0.38, 0.38),
            "distractor_bias": (-0.38, 0.38),
            "emotion_amplification_bias": (-0.38, 0.38), 
            "eta": (-1, 1),
            "eta_r": (-5, 5),
            "tau": (0.15, min_rt)  # Safe usage
        }

        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        # print(f"DEBUG: min RT in data = {min_rt}")  
        # print(f"DEBUG: set tau bounds = (0.15, {min_rt})")

        super().__init__(self.param_number, list(self.bounds.values()), self.parameter_names)

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, alpha_racial_bias, beta, characteristic_time, emotion_bias, 
                         racial_bias, distractor_bias, emotion_amplification_bias, eta, eta_r, tau, dt=DT, var=VAR, 
                         nTrials=NTRIALS, noiseseed=NOISESEED):
        """
        Simulates the Race-Emotion Bias Diffusion Model for Conflict (DMC).

        Parameters:
        - alpha: boundary separation
        - beta: initial bias
        - mu_c: drift rate of controlled process
        - shape: shape parameter for automatic activation time-course
        - characteristic_time: duration of automatic process
        - emotion_bias, racial_bias, distractor_bias: biases affecting drift rate
        - tau: non-decision time
        - dt: time step
        - var: variance of noise
        - nTrials: number of trials
        - noiseseed: seed for noise randomness
        """

        choicelist = np.full(nTrials, np.nan, dtype=np.float64)
        rtlist = np.full(nTrials, np.nan, dtype=np.float64)
        shape = 2
        # beta = 0.5

        np.random.seed(noiseseed)

        update_jitter = np.random.normal(loc=0, scale=var, size=1000)
        update_jitter = update_jitter * (10**eta)

        mu_c = 0.5
        shape = 2

        condition_list = np.zeros((nTrials, 3), dtype=np.int64)
        
        for i in range(nTrials // 8):
            condition_list[i] = [0, 0, 0]
            condition_list[i + nTrials // 8] = [0, 0, 1]
            condition_list[i + 2 * nTrials // 8] = [0, 1, 0]
            condition_list[i + 3 * nTrials // 8] = [0, 1, 1]
            condition_list[i + 4 * nTrials // 8] = [1, 0, 0]
            condition_list[i + 5 * nTrials // 8] = [1, 0, 1]
            condition_list[i + 6 * nTrials // 8] = [1, 1, 0]
            condition_list[i + 7 * nTrials // 8] = [1, 1, 1]
        
        # condition_str_list = [['0-0-0'] * (nTrials // 8) + ['0-0-1'] * (nTrials // 8) + ['0-1-0'] * (nTrials // 8) + ['0-1-1'] * (nTrials // 8) +
        #                     ['1-0-0'] * (nTrials // 8) + ['1-0-1'] * (nTrials // 8) + ['1-1-0'] * (nTrials // 8) + ['1-1-1'] * (nTrials // 8)] 
        
        for n in range(nTrials):
            caution = 0.5 * (1 + np.erf(-(alpha + alpha_racial_bias * condition_list[n, 1]) / np.sqrt(2))) * 0.38
            # neutral- if it is incongruent, you should have a further negative distraction bias 
            if condition_list[n,0] == 0:
                peak_amplitude = (-emotion_bias - racial_bias * condition_list[n, 1] - 
                                distractor_bias * condition_list[n, 2] - emotion_amplification_bias * condition_list[n, 2])
            # emotional - if it is congruent, you should have a further positive distractor bias 
            else:
                peak_amplitude = (emotion_bias + racial_bias * condition_list[n, 1] - 
                                distractor_bias * condition_list[n, 2] + emotion_amplification_bias * (1 - condition_list[n, 2]))
                beta += 2 * (0.5 - beta)
            # print('PEAK AMPLITUDE', peak_amplitude)

            t = tau
            evidence = beta * caution / 2 - (1 - beta) * caution / 2
            np.random.seed(n)
            # print('EVIDENCE', evidence)
            # print('ALPHA/2', alpha/2)

            while -caution / 2 < evidence < caution / 2:
                # if condition_list[n, 0] == 1:
                delta = ((peak_amplitude * np.exp(-(t / characteristic_time)) *
                        ((t * np.exp(1)) / ((shape - 1) * characteristic_time))**(shape - 1) * 
                        (((shape - 1) / t) - (1 / characteristic_time))) + mu_c)
                    # print('DELTA TOP', delta)
                # else:
                    # delta = ((-peak_amplitude * np.exp(-(t / characteristic_time)) *
                    #           ((t * np.exp(1)) / ((shape - 1) * characteristic_time))**(shape - 1) * 
                    #           (((shape - 1) / t) - (1 / characteristic_time))) + mu_c)
                    # print('DELTA BOTTOM', delta)

                delta_noise = np.random.choice(update_jitter)
                delta_noise = delta_noise*(np.exp(-1*(eta_r/2)*((t-tau))))
                evidence += delta*dt + delta_noise
                t += dt

                if evidence > caution / 2:
                    # print('BOUND')
                    choicelist[n] = 1
                    rtlist[n] = t
                    break  
                elif evidence < -caution / 2:
                    # print('NOT BOUND')
                    choicelist[n] = 0
                    rtlist[n] = t
                    break
        
        return np.arange(1, nTrials + 1), choicelist, rtlist, condition_list