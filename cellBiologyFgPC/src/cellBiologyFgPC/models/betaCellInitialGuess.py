r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    The class BetaCellInitModel is used to calculate the 
                initial guess for the FgPC model of the beta cell system.
"""
import argparse
import logging

import numpy as np

from fgPC.models.initialGuessFunctions import InitialGuessFunctions
from fgPC.utils.plotlib import Plotter, LinePlot

from cellBiologyFgPC.models.betaCellInt import integratedBetaCellSystem

class BetaCellInitModel(InitialGuessFunctions):
    r"""
    The class BetaCellInitModel is used to calculate the initial guess for the
    FgPC model of the beta cell system.
    """

    def __init__(self, sinMt):

        self.sinMt = sinMt # indicates if for the selfexcited system 
                           # the first sin is set to zero instead of 
                           # using var1_dot = 0
        super().__init__()

    def initialGuessFgPC(self,
                         h: int,
                         ngPC: int,
                         configFgPC: argparse.Namespace,
                         configInt: argparse.Namespace,
                         resultFolder: str,
                         systemStr: str,
                         tag: str,
                         logger: logging.Logger = None,
                         val: float = None):
        r"""
        Method to calculate the initial guess for the FgPC model of 
        the beta cell system.

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
        list of initialGuess and amp_s1
        initialGuess : np.ndarray
            Initial guess for the FgPC model
        amp_s1 : float
            Amplitude of the first sin function
        """
        
        # get deterministic value of uncertain variable
        if val is None:
            val = self.getUncertainValue(configFgPC, logger)

        # get intial guess for value
        strVal = '{:.3e}'.format(val)
        strFolder = resultFolder + "3_calculatedData/preCalcData/" 
        
        initialGuess, amp_s1 = self.calcAndGetData(tag, 
                                                   strVal,
                                                   val, 
                                                   configInt,
                                                   h,
                                                   ngPC, 
                                                   strFolder,
                                                   logger,
                                                   False)

        return [[initialGuess, amp_s1]]
    
    def getIntegrationData(self, 
                           systemStr: str, 
                           val: float, 
                           configInt: argparse.Namespace, 
                           logger: logging.Logger = None):
        r"""
        Method to calculate the integration data of the beta cell system.

        Parameters
        ----------
        systemStr : str
            Not neede in this system, but required for other objects.
        val : float
            Uncertain value
        configInt : argparse.Namespace
            Configuration parameters for the integration
        logger : logging.Logger (optional = None)
            Instance of the logger

        Returns
        ----------
        intSol : np.array
            Integration data
        intTime : np.array
            Time steps
        """
        
        intSol, intTime = self.getIntegrationDataElectric(val, configInt, logger)

        if logger is not None:
            logger.info("Integration data generated.")

        return intSol, intTime/1000
    
    def getIntegrationDataElectric(self, 
                                   val: float, 
                                   configInt: argparse.Namespace, 
                                   logger: logging.Logger = None):
        r"""
        Method to calculate the integration data of the beta cell system.

        Parameters
        ----------
        val : float
            Uncertain value
        configInt : argparse.Namespace
            Configuration parameters for the integration
        logger : logging.Logger (optional = None)
            Instance of the logger

        Returns
        ----------
        intDataPer : list
            Integration data with only complete oscillations
            with each variable data in a seperated list entry
        intTimePer : list
            Time data with only complete oscillations
        """

        timIntBetaCell = integratedBetaCellSystem(configInt, logger)

        timIntBetaCell.cellModel.atp = val
        tList = [configInt.t0, configInt.t_end_tr* 60000, configInt.t_step_tr]

        initVals = [configInt.V_0, configInt.n_0, configInt.Ca_0]

        intData , intTime = timIntBetaCell.integrationModel(initVals,
                                                            tList)
        
        intDataTrunc, intTimeTrunc = self.getPeriod(intData, intTime)

        if intDataTrunc is False:
            intDataPer = intData
            intTimePer = intTime
        else:
            intDataPer = intDataTrunc
            intTimePer = intTimeTrunc
            for j in range(10):
                intDataPer = np.concatenate((intDataPer, intDataPer), axis=1)
                intTimePer = np.concatenate((intTimePer, intTimePer+intTimePer[-1]+intTimePer[1]-intTimePer[0]))
        
        return intDataPer, intTimePer        

    def getPeriod(self,
                  intData: np.ndarray, 
                  intTime: np.ndarray):
        r""" Methods extracts only complete oscillations
        from the integration data, which are later used for the FFT
        
        Parameters
        ----------
        intData : np.array
            Integration data
        intTime : np.array
            Time data
            
        Returns
        -------
        intDataPer : list
            Integration data with only complete oscillations
            with each variable data in a seperated list entry
        intTimePer : list
            Time data with only complete oscillations
        """

        persol = False
        targetVals = np.array([sublist[-1] for sublist in intData])
        for i in range(len(intData[0])-100, 0, -1):
            elc_cur = np.array([sublist[i-1] for sublist in intData])
            
            if all(np.abs(elc_cur-targetVals)/np.abs(targetVals) < 1e-3):
                intDataPer = np.array([sublist[i:] for sublist in intData])
                intTimePer = intTime[i:]-intTime[i]
                persol = True
                break
        
        if not persol:
            intDataPer = False
            intTimePer = False
        
        return intDataPer, intTimePer

    def getInitialGuess(self, 
                        h: int, 
                        ngPC: int, 
                        peakDataList: list, 
                        logger: logging.Logger = None):
        r"""
        Method creates the initial guess for the FgPC model from 
        the peak data of the FFT.

        Parameters
        ----------
        h : int
            Number of harmonics
        ngPC : int
            Degreee of the gPC expansion
        peakDataList : list
            List with the peak data, 
            first frequencies, ampltidues and norm amplitudes
        logger : Logger (optional = None)
            Instance of the logger

        Returns
        ----------
        initialGuess : np.array
            Initial guess for the FgPC model
        v_sin1 : float
            Amplitude of the first sin function
        """

        i = 0
        for peakData in peakDataList:
            valAmp = peakData[1]
            valH = int((valAmp.shape[0]-1)/2)
            if i == 0:
                if peakData[0].shape[0] == 1:
                    baseFreq = peakData[0][0]
                else:
                    baseFreq = peakData[0][1]
                if valH > h:
                    if self.sinMt == 1:
                        valAmpNew = np.concatenate((valAmp[0:h+1], valAmp[valH+2:valH+1+h]))
                        v_sin1 = valAmp[valH+1]
                    else:
                        valAmpNew = np.concatenate((valAmp[0:h+1], valAmp[valH+1:valH+1+h]))
                        v_sin1 = 0
                else:
                    if self.sinMt == 1:
                        valAmpNew = np.zeros(2*h)
                        if valH == 0:
                            valAmpNew[0] = valAmp[0]
                            v_sin1 = 0
                        else:
                            valAmpNew[0:valH+1] = valAmp[0:valH+1]
                            valAmpNew[h+1:h+valH] = valAmp[valH+2:]
                            v_sin1 = valAmp[valH+1]
                    else:
                        valAmpNew = np.zeros(2*h+1)
                        valAmpNew[0:valH+1] = valAmp[0:valH+1]
                        valAmpNew[h+1:h+1+valH] = valAmp[valH+1:]
                        v_sin1 = 0
            
                guessVec = valAmpNew
            else:
                if valH > h:
                    valAmpNew = np.concatenate((valAmp[0:h+1], valAmp[valH+1:valH+1+h]))
                else:
                    valAmpNew = np.zeros(2*h+1)
                    valAmpNew[0:valH+1] = valAmp[0:valH+1]
                    valAmpNew[h+1:h+1+valH] = valAmp[valH+1:]

                guessVec = np.concatenate((guessVec, valAmpNew))
            i += 1

        guessVec = np.concatenate((guessVec, np.array([baseFreq])))

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

        return initialGuess, v_sin1
    