r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    Class with the integration model of the Duffing oscillator.

"""

import numpy as np

from scipy.integrate import solve_ivp

class integratedDuffing:

    def __init__(self, config, logger):
        self.config = config
        self.alpha = config.alpha
        self.beta = config.beta
        self.delta = config.delta
        self.gamma = config.gamma
        self.omega = config.omega
        self.logger = logger

    def integrationModel(self, 
                         initialVals: np.ndarray, 
                         tList: list):
        r"""
        Method to integrate the Duffing model

        Parameters
        ----------
        initialVals : np.ndarray
            Initial values for the integration
        tList : list
            List with start and end time of the integration
            and desired time steps

        Returns
        ----------
        vals : list
            List with the values of the variables
        sol.t : np.ndarray
            List with the time steps
        """

        sol = solve_ivp(self.duffingModel, 
                        [tList[0], tList[1]], initialVals, 
                        t_eval = np.linspace(tList[0], tList[1], tList[2]))
        

        x = sol.y[0]
        v = sol.y[1]

        return [x, v], sol.t
    
    def duffingModel(self, 
                     t: float, 
                     y: np.ndarray):
        r"""
        Method that return the duffing model equations in
        first order form
        
        Parameters
        ----------
        t : float
            Time
        y : list
            List with the state variables

        Returns
        ---------
        list
            List with the first order equations
        """

        x = y[0]
        v = y[1]

        dxdt = v
        dvdt = - self.delta*v - self.alpha*x - self.beta*x**3 + self.gamma* np.cos(self.omega*t)

        return [dxdt, dvdt]
