r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    The class DuffingInitModel is used to calculate the 
                initial guess for the FgPC model of the Duffing system.
"""
import os 
import pickle
import argparse
import logging

import numpy as np

from fgPC.models.initialGuessFunctions import InitialGuessFunctions
from fgPC.utils.plotlib import Plotter, LinePlot

from duffingFgPC.models.duffingInt import integratedDuffing

class DuffingInitModel(InitialGuessFunctions):
    r"""
    The class DuffingInitModel is used to calculate the initial guess for the
    FgPC model of the Duffing system.
    """

    def __init__(self):
        super().__init__()

    def initialGuessFgPC(self,
                         h: int,
                         ngPC: int,
                         configFgPC: argparse.Namespace,
                         configInt: argparse.Namespace,
                         resultFolder: str,
                         systemStr: str,
                         tagIn: str,
                         logger: logging.Logger = None,
                         val: float = None):
        r"""
        Method to calculate the initial guess for the FgPC model of 
        the Duffing oscilllator.

        Parameters
        ----------
        h : int
            number of harmonics
        ngPC : int
            degree of polynomials
        configFgPC : argparse.Namespace
            configuration parameters for the FgPC model
        configInt : argparse.Namespace
            configuration parameters for the integration
        resultFolder : str
            folder where the results are stored
        systemStr : str
            name of the system
        tagIn : str
            solution tag
        logger : Logger (optional = None)
            instance of the logger

        Returns
        ----------
        initialGuess : np.array
            Initial guess for the FgPC model
        """
        
        # get deterministic value of uncertain variable
        if val is None:
            val = self.getUncertainValue(configFgPC, logger)

        if tagIn == "deflation":
            tagList = [ "unstable", "small", "large"]
        else:
            tagList = [tagIn]

        strVal = '{:.1e}'.format(val)
        strFolder = resultFolder + "2_calculatedData/preCalcData/" 

        initialGuessList = []
        for tag in tagList:
            
            if tag != "unstable":

                if tag == "large":
                    configInt.x0 = 2
                    configInt.xt0 = 2
                else:
                    configInt.x0 = 0.1
                    configInt.xt0 = 0.1

                initialGuess = self.calcAndGetData(tag, 
                                                   strVal, 
                                                   val, 
                                                   configInt,
                                                   h,
                                                   ngPC, 
                                                   strFolder,
                                                   logger)
                
            else:
                initialGuess = self.getUnstableGuess(h, ngPC)

            initialGuessList.append(initialGuess)

        return initialGuessList

    def getIntegrationData(self, 
                           systemStr: str, 
                           val: float, 
                           configInt: argparse.Namespace, 
                           logger: logging.Logger = None):
        r"""
        Method to calculate the integration data for the Duffing system

        Parameters
        ----------
        systemStr : str
            name of the system
        val : float
            value of the uncertain variable
        configInt : argparse.Namespace
            configuration parameters for the integration
        logger : Logger (default = None)
            instance of the logger
        
        Returns
        ----------
        intData_ss : np.ndarray
            integration data for the steady state solution
        intTime_ss : np.ndarray
            time steps for the steady state solution
        """
        
        timIntDuffing = integratedDuffing(configInt, logger)

        timIntDuffing.alpha = val

        tE = 2*np.pi/configInt.omega*configInt.tE
        t0 = configInt.t0
        dt = configInt.dt
        if systemStr == "unstable":
            tList = [tE, t0, int((tE-t0)/dt)]
        else:
            tList = [t0, tE, int((tE-t0)/dt)]

        initVals = [configInt.x0, configInt.xt0]

        # transient integration
        intData_tr , intTime_tr = timIntDuffing.integrationModel(initVals,
                                                                 tList)
        
        t0_ss = intTime_tr[-1]
        tE_ss = t0_ss + 2*np.pi/configInt.omega*configInt.tE/2
        dt_ss = int((tE_ss-t0_ss)/dt)
        tList_ss = [t0_ss, tE_ss, dt_ss]
        initVals_ss = [intData_tr[0][-1], intData_tr[1][-1]]
        intData_ss , intTime_ss = timIntDuffing.integrationModel(initVals_ss,
                                                                 tList_ss)
        
        if logger is not None:
            logger.info("Integration data generated.")

        return intData_ss , intTime_ss
    
    def getInitialGuess(self, 
                        h: int, 
                        ngPC: int, 
                        peakDataList: list, 
                        logger: logging.Logger = None):
        r"""
        Method to calculate the initial guess for the Duffing FgPC model

        Parameters
        ----------
        h : int
            number of harmonics
        ngPC : int
            degree of polynomials
        peakDataList : list
            list with the peak data of the FFT
        logger : Logger (default = None)
            instance of the logger

        Returns
        ----------
        initialGuess : np.ndarray
            initial guess for the solution
        """

        guessVec = np.array([])
        peakData = peakDataList[0]

        valAmp = peakData[1]
        valH = int((valAmp.shape[0]-1)/2)
        valCos = valAmp[1:valH+1]
        valSin = valAmp[valH+1:]

        if valH > 0:
            freq = peakData[0][1:]/peakData[0][1]
        
        vecCos = np.zeros(h)
        vecSin = np.zeros(h)
        freqIdx = 0
        for idxh in range(h):
            if freqIdx <= valH-1:
                if int(freq[freqIdx]) == idxh+1:
                    vecCos[idxh] = valCos[freqIdx]
                    vecSin[idxh] = valSin[freqIdx]
                    freqIdx += 1

        vec = np.concatenate((np.array([valAmp[0]]),vecCos, vecSin))

        guessVec = np.concatenate((guessVec, vec))
            

        if ngPC > 1:
            newLen = guessVec.shape[0]*(ngPC)
            initialGuess = np.zeros(newLen)
            j = 0
            for i in range(guessVec.shape[0]):
                initialGuess[j:j+ngPC] = self.gPCValCalc(guessVec[i], ngPC)
                j += ngPC
        else:
            initialGuess = guessVec

        if logger is not None:
            logger.info("Initial guess generated.")

        return initialGuess
    
    def getUnstableGuess(self,
                         h: int,
                         ngPC: int):
        r"""
        Method to calculate the initial guess for the unstable solution
        
        Parameters
        ----------
        h : int
            number of harmonics
        ngPC : int
            degree of polynomials        
            
        Returns
        ----------
        initialGuess : np.array
            initial guess for the unstable solution
        """
        cos1 = -0.836
        sin1 = 0.588

        cos3 = 0.00517
        sin3 = 0.0169

        cosVec = np.zeros(3)
        cosVec[0] = cos1
        cosVec[2] = cos3
        sinVec = np.zeros(3)
        sinVec[0] = sin1
        sinVec[2] = sin3 
        
        cosVecTarg = np.zeros(h)
        sinVecTarg = np.zeros(h)
        for idxH in range(h):
            if idxH < 3:
                cosVecTarg[idxH] = cosVec[idxH]
                sinVecTarg[idxH] = sinVec[idxH]

        initGuessH = np.concatenate((np.array([0]), cosVecTarg, sinVecTarg))

        initialGuess = np.zeros((2*h+1)*ngPC)
        i = 0
        for idxVal in range(len(initGuessH)):
            gPCVec = np.zeros(ngPC)
            for j in range(ngPC):
                gPCVec[j] = initGuessH[idxVal]*10**(-j)
            initialGuess[i:i+ngPC] = gPCVec
            i += ngPC

        return initialGuess