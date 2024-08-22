r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    The superclass InitialGuessFunctions is used to calculate the
                initial guess for a given FgPC model. Needsto be specified
                for each system.

"""
import os
import pickle
import argparse
import logging

import numpy as np

from abc import ABC, abstractmethod
from fgPC.utils.fftmodel import FFTModel

class InitialGuessFunctions(ABC):
    r"""
    The superclass InitialGuessFunctions is used to calculate the
    initial guess for a given FgPC model. Needsto be specified.
    """

    def __init__(self):
        pass

    @abstractmethod
    def initialGuessFgPC(self):
        pass

    @abstractmethod
    def getIntegrationData(self):
        pass

    @abstractmethod
    def getInitialGuess(self):
        pass

    def getUncertainValue(self, 
                          configHBgPC: argparse.Namespace, 
                          logger: logging.Logger = None):
        r""" Method calculates middel of the uncertain value 

        Parameters
        ----------
        configHBgPC : argparse.Namespace
            configuration parameters for the HBgPC model
        logger : logging.Logger
            instance of the logger
        
        Returns
        ----------
        val : float
            uncertain value
        """

        val = (configHBgPC.high-configHBgPC.low)/2 + configHBgPC.low
        
        if logger is not None:
            logger.info("Uncertain value for integration set to: " + '{:.2e}'.format(val))
        
        return val


    def calcAndGetData(self, 
                       tag: str, 
                       strVal: str, 
                       val: float, 
                       configInt: argparse.Namespace, 
                       h: int,
                       ngPC: int, 
                       strFolder:str, 
                       logger: logging.Logger = None, 
                       forced: bool = True):
        r""" Method calculates the initial guess of the HBgPC model.
        Along with the integration data and the fft data.

        Parameters
        ----------
        tag : str
            solution tag
        strVal : str
            string representation of the value
        val : float
            uncertain value
        configInt : argparse.Namespace
            configuration parameters for the integration
        h : int
            number of harmonics
        ngPC : int
            degree of polynomials
        strFolder : str
            folder where the data is stored
        logger : logging.Logger
            instance of the logger
        forced : bool
            boolean to indicate if the system is forced

        Returns
        ----------
        initialGuess : np.array
            initial guess for the solution
        amp_s1 : float, when forced is False
            amplitude of the first sin function
        """

        strInt = tag + "_var" + strVal + "_1intData.pkl"
        if os.path.exists(strFolder + strInt):
            with open(strFolder + strInt, "rb") as f:
                intData = pickle.load(f)
            intSol = intData[0]
            intTime = intData[1]
        else:
            intSol, intTime = self.getIntegrationData(tag, val, configInt, logger)
            with open(strFolder + strInt, "wb") as f:
                pickle.dump([intSol, intTime], f)

            
        _, peakData = self.getFFTData(intSol, 
                                      intTime,
                                      configInt.fftMinVal,
                                      configInt.fftHighFreq,
                                      configInt.fftDist,
                                      logger)
        
        if forced:
            initialGuess = self.getInitialGuess(h,
                                                ngPC,
                                                peakData,
                                                logger)
            
            return initialGuess
        else:
            initialGuess, amp_s1 = self.getInitialGuess(h,
                                                        ngPC,
                                                        peakData,
                                                        logger)
            
            return initialGuess, amp_s1


    def getFFTData(self, 
                   intSol: list, 
                   intTime: np.ndarray, 
                   thresHold: float, 
                   highFreq: float, 
                   dist: float = None, 
                   logger: logging.Logger = None):
        r""" Method calculates the FFT data for the given solution

        Parameters
        ----------
        intSol : list
            list with the integration solutions for each variable
        intTime : np.ndarray
            list with the time points
        thresHold : float
            threshold for the FFT, when to consider a peak
        highFreq : float
            highest frequency to be considered
        dist : float, optional
            distance in points between the peaks
        logger : logging.Logger (default = None)
            instance of the logger

        Returns
        ----------
        fftData : list
            list with the FFT data, with the following structure:
            [frequencies, norm of amplitudes, amplitudes]
        peak : list
            list with the peak data, with the following structure:
            [peak frequencies, peak amplitudes, norm of peak amplitudes]
        """

        
        myFFT = FFTModel(thresHold, dist)
        fftData = []
        peak = []
        phaseshift = None
        for valSol in intSol:
            fftDataVal, peakVal, phaseshift = \
                myFFT.performFFT(valSol, 
                                intTime, 
                                highFreq=highFreq,
                                phaseshift=phaseshift)
            fftData.append(fftDataVal)
            peak.append(peakVal)

        if logger is not None:
            logger.info("FFT data generated.")

        return fftData, peak
    
    def gPCValCalc(self, 
                   val: float, 
                   ngPC: int):
        r""" Method calculates the gPC values for a given value
        
        Parameters
        ----------
        val : float
            Value for which the gPC values are calculated
        ngPC : int
            Number of gPC values
            
        Returns
        -------
        gPCVal : np.array
            gPC values
        """

        gPCVal = np.zeros(ngPC)
        gPCVal[0] = val
        
        return gPCVal
    