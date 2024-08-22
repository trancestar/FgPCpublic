r"""
Author:         Lars de Jong
Date:           2024-05-15
Description:    Class with the integration model of the beta cell.

"""
import argparse
import logging

import numpy as np

from scipy.integrate import solve_ivp

from cellBiologyFgPC.models.betaCellModel import betaCellfunctions

class integratedBetaCellSystem:
    r"""
    Class with the integration model of the beta cell.
    """

    def __init__(self, 
                 config: argparse.Namespace, 
                 logger: logging.Logger = None) -> None:
        
        self.logger = logger

        self.cellModel = betaCellfunctions()

        # Read config file
        self.V_0 = config.V_0
        self.n_0 = config.n_0
        self.Ca_0 = config.Ca_0
        
    def integrationModel(self, 
                         initialVals: np.ndarray, 
                         tList: list):
        r"""
        Method handles integration of the beta cell model

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
        sol = solve_ivp(self.betaCellSystemFunction, 
                        [tList[0], tList[1]], initialVals,
                        t_eval = np.arange(tList[0],tList[1],tList[2]), 
                        method='Radau')
        
        v = sol.y[0,]
        n = sol.y[1,]
        ca = sol.y[2,]
        
        return [v, n, ca], sol.t
             

    def betaCellSystemFunction(self, 
                               t: float, 
                               vals: np.ndarray):
        r"""
        Method contains the first order functions of the 
        integrated oscilator model

        Parameters
        ----------
        t : float
            time in ms
        vals : np.ndarray
            list with the values of the variables

        Returns
        ----------
        vals :  list
            list with the first order equations of 
            corresponding beta cell system
        """

        v = vals[0]
        n = vals[1]
        ca = vals[2]

        ca_er = self.cellModel.Ca_er
        atp = self.cellModel.atp
        adp = self.cellModel.concentrationADP(atp)
        
        i_ca, i_k, i_kca, i_katp = \
            self.cellModel.calculateIonCurrent(v, n, ca, adp, atp)

        n_inf = self.cellModel.activationFun(v, self.cellModel.n_in, \
                        self.cellModel.s_n)
        
        j_mem = self.cellModel.calcJmem(i_ca, ca)
        j_er = self.cellModel.calcJer(ca, ca_er)

        dvdt = -1/self.cellModel.C_mem * (i_ca + i_k + i_kca + i_katp)
        dndt = (n_inf-n)/self.cellModel.tau_n
        dcadt = self.cellModel.f_Ca*(j_mem-j_er)

        return [dvdt, dndt, dcadt]