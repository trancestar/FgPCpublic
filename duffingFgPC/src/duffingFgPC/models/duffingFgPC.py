r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    FgPC class of the Duffing oscillator.

"""
import time
import argparse
import logging

import numpy as np

from fgPC.models.fgpc import fouriergenPolynomialChaos as FgPC

class duffingFgPC(FgPC):
    r"""
    This class creates the FgPC model object for the Duffing oscillator.
    """

    def __init__(self, configFgPC: argparse.Namespace,
                 configInt: argparse.Namespace,
                 amp_s1: float = 0,
                 deflation: bool = False,
                 logger: logging.Logger = None,
                 **kwargs):
        r"""
        Constructor of the duffingFgPC class.

        Parameters
        ----------
        configFgPC : argparse.Namespace
            Configuration for the FgPC model.
        configInt : argparse.Namespace
            Configuration for the integration model.
        amp_s1 : float (default = 0)
            Sine amplitude of the first harmonic.
        deflation : bool (default = False)
            Boolean to indicate if deflation is used.
        logger : logging.Logger (default = None)
            The logger object to log messages.
        **kwargs : dict
            Additional key word arguments.
            quadRule : str
                rule for quadrature. For options see chaospy generate_quadrature.
            tol : float
                tolerance for root finding.
        """
        super().__init__([configFgPC.H], 
                         configFgPC.nPt,
                         configFgPC.ngPC,
                         configFgPC.nQPt,
                         configFgPC.distStr,
                         configFgPC.low,
                         configFgPC.high,
                         amp_s1,
                         configFgPC,
                         logger,
                         deflation,
                         **kwargs)

        self.alpha = configInt.alpha
        self.beta = configInt.beta
        self.delta = configInt.delta
        self.gamma = configInt.gamma
        self.omega = configInt.omega

        self.extForcVec = self.calcExtForceVec(configFgPC.H,configInt.gamma)

        if self.logger is not None:
            self.logger.info("Duffing FgPC model initialized")

    @FgPC.harmonics.setter
    def harmonics(self, value):
        super(duffingFgPC, type(self)).harmonics.__set__(self, value)
        self.extForcVec = self.calcExtForceVec(self.harmonics[0],self.gamma)

    def constructCosSinFgPCCoeffs(self):
        r"not needed for the Duffing oscillator."
        pass

    def calculateFgPC(self, 
                      currentGuess: np.ndarray,
                      quadPt: np.ndarray, 
                      polyVals: np.ndarray):
        r"""
        Calculates the time residuum of the Duffing oscillator.

        Parameters
        ----------
        currentGuess : np.ndarray
            Current guess for the FgPC coefficients.
        quadPt : np.ndarray
            quadrature point
        polyVals : np.ndarray
            Polynomial values at quadrature point.
        
        Returns
        ----------
        timeResiduum : np.ndarray
            Time residuum of the Duffing oscillator.
        """

        pos, vel, acc = self.calculatePosVelAcc(currentGuess, polyVals)

        timeResiduum = self.calculateTimeResiduum(pos, vel, acc, quadPt)

        return timeResiduum
    
    def calculateHB(self, 
                    currentGuess: np.ndarray, 
                    sample: float):
        r"""
        Calculates the HB solution of the Duffing oscillator.

        Parameters
        ----------
        currentGuess : np.ndarray
            Current guess for the FgPC coefficients.
        sample : float
            current sample

        Returns
        ----------
        residuum : np.ndarray
            Residuum of the Duffing oscillator.
        """

        t0_total = time.time()
        hbCoeffs_c = self.convertAllHBSinCos2Complx(currentGuess)

        pos = self.calcPosition(hbCoeffs_c,self.E_nh_c_total)
        vel = self.calcVelocity(hbCoeffs_c,self.E_nh_c_total, 
                                self.omega, self.derMat_c_total)
        acc = self.calcAcceleration(hbCoeffs_c,self.E_nh_c_total, 
                                   self.omega, self.derMat_c_total)

        timeResiduum = self.calculateTimeResiduum(pos, vel, acc, sample)

        residuum = self.calcFreqResiduum(timeResiduum)

        t1_total = time.time()
        self.itrCt += 1
        print("Iteration: " + str(self.itrCt) + 
              " Function Eval Time: " + '{:.8f}'.format(t1_total-t0_total) +
              " Residuum Norm: " + '{:.6e}'.format(np.sqrt(np.sum(residuum**2))), end='\r')
        return residuum

    def calculatePosVelAcc(self, 
                           currentGuess: np.ndarray,
                           polyVals: np.ndarray):
        r"""
        Calculates the position, velocity and acceleration of the Duffing oscillator.

        Parameters
        ----------
        currentGuess : np.ndarray
            Current guess for the FgPC coefficients.
        polyVals : np.ndarray
            Polynomial values at quadrature point.

        Returns
        ----------
        pos : np.ndarray
            Position of the Duffing oscillator over one period
        vel : np.ndarray
            Velocity of the Duffing oscillator over one period
        acc : np.ndarray
            Acceleration of the Duffing oscillator over one period
        """

        ident = np.eye(self.totalH)
        hbCoeffs = np.dot(np.kron(ident,polyVals), currentGuess)

        hbCoeffs_c = self.convertAllHBSinCos2Complx(hbCoeffs)

        pos = self.calcPosition(hbCoeffs_c,self.E_nh_c_total)
        vel = self.calcVelocity(hbCoeffs_c,self.E_nh_c_total, 
                                self.omega, self.derMat_c_total)
        acc = self.calcAcceleration(hbCoeffs_c,self.E_nh_c_total, 
                                   self.omega, self.derMat_c_total)

        return pos, vel, acc
    
    def calculateTimeResiduum(self, 
                              pos: np.ndarray,
                              vel: np.ndarray,
                              acc: np.ndarray,
                              quadPt: np.ndarray):
        r"""
        Calculates the time residuum of the Duffing oscillator.

        Parameters
        ----------
        pos : np.ndarray
            Position of the Duffing oscillator.
        vel : np.ndarray
            Velocity of the Duffing oscillator.
        acc : np.ndarray
            Acceleration of the Duffing oscillator.
        quadPt : np.ndarray
            quadrature point.

        Returns
        ----------
        timeResiduum : np.ndarray
            Time residuum of the Duffing oscillator over one period.
        """

        linTerm = acc + self.delta * vel + quadPt * pos
        nonlinTerm = self.beta * pos**3
        extForce = np.real(self.E_nh_c_total @ self.extForcVec)

        timeResiduum = linTerm + nonlinTerm - extForce

        return timeResiduum
    
    def calcExtForceVec(self, 
                        H: int, 
                        gamma: float):
        r"""
        Calculates the external force vector.

        Parameters
        ----------
        H : int
            Number of harmonics.
        gamma : float
            Force amplitude.

        Returns
        ----------
        extForceVec : np.ndarray
            External force vector.
        """

        extForceVec = np.zeros(2*H+1)
        extForceVec[H-1] = gamma/2
        extForceVec[H+1] = gamma/2

        return extForceVec