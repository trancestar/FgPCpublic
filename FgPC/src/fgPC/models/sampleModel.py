r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    This class handles the sampling of the MC and FgPC models. 
                
"""
import argparse
import logging
import os
import pickle
import time
import copy

import numpy as np
import chaospy as cp

from scipy.optimize import root

from fgPC.models.fgpc import fouriergenPolynomialChaos as FgPC
from fgPC.models.initialGuessFunctions import InitialGuessFunctions as InitGuess


class samplingModel():
    r"""
    Class to handle the sampling of the MC and FgPC models.
    """

    def __init__(self,
                 configFgPC: argparse.Namespace,
                 sampleStr: str,
                 logger: logging.Logger = None):
        r"""
        Constructor method of sampling class.

        Parameters
        ----------
        configFgPC : argparse.Namespace
            The configuration object containg all input parameters.
        sampleStr : str
            The string to store the samples.
        logger : logging.Logger (default = None)
            The logger object to log messages.
        """
        self.logger = logger

        self.distStr = configFgPC.distStr
        self.getDistribution(self.distStr, 
                             configFgPC.low, 
                             configFgPC.high)

        self.sampleNr = configFgPC.sampleNr
        self.sampleStr = sampleStr
        self.generatSamples()

        self.mcUp = True

        if self.logger is not None:
            self.logger.info("Sampling model initialized.")

    def generatSamples(self):
        r"""
        Checks first if samples already exist. 
        Otherwise generates and saves them"""

        if not os.path.exists(self.sampleStr):
            samples = np.sort(self.dist.sample(self.sampleNr))
            with open(self.sampleStr, "wb") as f:
                pickle.dump(samples, f)            

    def getDistribution(self, distStr, low, high):
        r"""
        Method returns the distribution
        for the given distribution string

        Parameters
        ----------
        distStr : str
            Distribution string
        low : float
            either lower bound or mean of the distribution
        high : float
            either upper bound or standard deviation of the distribution
        """ 
        
        if distStr == "uniform" or distStr == "normal" or \
            distStr == "lognormal":
            if distStr == "uniform":
                self.dist = cp.Uniform(low, high)    
            elif distStr == "normal":
                self.dist = cp.Normal(low, high)
            elif distStr == "lognormal":
                self.dist = cp.LogNormal(low, high)
            self.lowLimit = self.dist.lower[0]
            self.highLimit = self.dist.upper[0]
        elif distStr == "beta":
            self.lowLimit = low
            self.highLimit = high
            self.dist = cp.Beta(5,5,low, high)
        else:
            raise ValueError("Distribution not implemented.")
        self.pdf = self.dist.pdf

    def calculateMC(self, 
                    afthbModel: FgPC, 
                    initialGuess: np.ndarray,
                    coeffs: np.ndarray = None, 
                    samplingIdx: int = 0):
        r"""
        Method calculates the Monte Carlo model

        Parameters
        ----------
        afthbModel : FgPC object
            instance of the FgPC model with the HBx method
        initialGuess : np.ndarray
            initial guess for the first sample
        coeffs : np.ndarray
            array with all corresponding HB coefficients 
            for the MC sampling
        samplingIdx : int
            index of the last evaluated sample

        Returns
        ----------
        results : np.ndarray
            list with the results of the Monte Carlo model
        samplingBool : bool
            boolean to indicate if the sampling was successful
        samplingIdx : int
            index of the last evaluated sample
        sample : float
            last evaluated sample
        """

        # load samples
        with open(self.sampleStr, "rb") as f:
            samples = pickle.load(f)
        
        if coeffs is None:
            coeffs = np.zeros((len(samples),initialGuess.shape[0]))
            
        samplingBool = True
        if samplingIdx == 0:
            self.initGuessMCMid = initialGuess
        if samplingIdx > int(len(samples)/2)-1:
            j = int(len(samples)/2) - (samplingIdx - (int(len(samples)/2)-1))
        else:
            j = int(len(samples)/2)+samplingIdx
        for i in range(samplingIdx,len(samples)):

            sample = samples[j]
            afthbModel.itrCt = 0
            result = root(afthbModel.calculateHB,
                          initialGuess,
                          args = (sample,))
            if not result.success:
                samplingBool = False
                break
            coeffs[j,:] = result.x

            initialGuess = result.x
            samplingIdx += 1

            if j == len(samples)-1:
                self.mcUp = False
                j = int(len(samples)/2)
                initialGuess = self.initGuessMCMid
            if self.mcUp:
                j += 1
            else:
                j -= 1

            perc = i/len(samples)*100
            if np.mod(perc, 5) == 0:
                print("\n" +str(perc) + r" % of MC samples done")
                if self.logger is not None:
                    self.logger.info(str(perc) + r" % of MC samples done")

        return coeffs, samplingBool, samplingIdx, sample

    def calculateFgPC(self, 
                      FgPCModel: FgPC, 
                      FgPCSol: np.ndarray):
        r"""
        Method evaluates the FgPC model at the given samples

        Parameters
        ----------
        FgPCModel : FgPC object
            instance of the FgPC model
        FgPCSol : np.ndarray
            Solution of the FgPC model
        
        Returns
        ----------
        coeffs : np.ndarray
            array with alll hb coefficients for all samples
        """
        # load samples
        with open(self.sampleStr, "rb") as f:
            samples = pickle.load(f)

        coeffsNr = int(FgPCSol.shape[0]/FgPCModel.ngPC)
        ident = np.eye(coeffsNr)

        coeffs = np.zeros((len(samples),coeffsNr))

        for sample, i in zip(samples,range(len(samples))):
            polyVals = FgPCModel.calculatePolynomials(sample)
            hbCoeffs = np.dot(np.kron(ident,polyVals), FgPCSol)

            coeffs[i,:] = hbCoeffs

            perc = i/len(samples)*100
            if np.mod(perc,5) == 0:
                print(str(perc) + r" % of FgPC samples done")
                if self.logger is not None:
                    self.logger.info(str(perc) + r" % of FgPC samples done")

        return coeffs
    
    def samplingOfModels(self,
                         FgPCSolList: list,
                         myFgPCmodel: FgPC,
                         tagList: list,
                         configFgPC: argparse.Namespace,
                         configInt: argparse.Namespace,
                         initModel: InitGuess,
                         resultFolder: str,
                         systemStr: str,
                         forced: bool = False,
                         logger: logging.Logger = None):
        r"""
        Method to sample the FgPC and MC models

        Parameters
        ----------
        FgPCSol : np.ndarray
            Solution of the FgPC model
        myFgPCmodel : FgPC object
            instance of the FgPC model
        tagList : list
            list with the tags of the solutions
        configFgPC : argparse.Namespace
            configuration parameters for the FgPC model
        configInt : argparse.Namespace
            configuration parameters for the integration
        initModel : InitialGuessFunctions object
            instance of the initial guess model
        resultFolder : str
            folder where the results are stored
        systemStr : str
            name of the system
        forced : bool
            boolean to indicate if the system is forced
        logger : logging.Logger (default = None)
            instance of the logger

        Returns
        ----------
        FgPCResultList : list
            list with the sampling of the FgPC model for each solution in FgPCSol
        mcResultList : list
            list with the sampling of the Monte Carlo model for each solution in tagList
        """


        # --- gPC sampling ---

        FgPCResultList = []
        
        for curFgPCsol in FgPCSolList:

            if forced:
                FgPCsol = curFgPCsol
            else:
                FgPCsol = curFgPCsol[0]
                myFgPCmodel.amp_s1 = curFgPCsol[1]

            # FgPC calculations
            t0_FgPC = time.time()
            FgPCResults = self.calculateFgPC(myFgPCmodel,
                                                FgPCsol)
            FgPCResultList.append(FgPCResults)
            
            t1_FgPC = time.time()
            if self.logger is not None:
                self.logger.info("FgPC sampling took: " + '{:.2f}'.format(t1_FgPC-t0_FgPC) + "s")
            print("FgPC sampling took: " + str(t1_FgPC-t0_FgPC) + "s")

        # --- Monte Carlo ---
        mcResultList = []
        for curTag in tagList:

            configHBMC = copy.deepcopy(configFgPC)
            configHBMC.ngPC = 1
            samplingBool = False
            samplingIdx = 0
            mcResults = None
            uqVal = None
            t0_mc = time.time()
            i = 0
            while not samplingBool:

                initialGuessMSList = \
                    initModel.initialGuessFgPC(configHBMC.H,
                                            configHBMC.ngPC,
                                            configHBMC,
                                            configInt,
                                            resultFolder,
                                            systemStr,
                                            curTag,
                                            logger,
                                            val = uqVal)
                if forced:
                    initialGuessMS = initialGuessMSList[0]
                else:
                    initialGuessMS = initialGuessMSList[0][0]
                    ampMS = initialGuessMSList[0][1]
                    if i < 2:
                        myFgPCmodel.amp_s1 = ampMS
                    elif i == 2:
                        myFgPCmodel.amp_s1 = 0
                        if self.logger is not None:
                            self.logger.info("Changed amp_s1 to 0 for sample index " + str(samplingIdx))
                    else:
                        raise ValueError("Solution for sample index " + str(samplingIdx) + " could not be found!")
                        
                    i += 1
                
                # mc sampling
                mcResults, samplingBool, samplingIdx, sample = \
                                self.calculateMC(myFgPCmodel,
                                                initialGuessMS,
                                                mcResults, 
                                                samplingIdx)
                
                uqVal = sample
            
            t1_mc = time.time()
            if self.logger is not None:
                self.logger.info("MC sampling took: " + '{:.2f}'.format(t1_mc-t0_mc) + "s")
            print("\n MC sampling took: " + str(t1_mc-t0_mc) + "s")
            
            mcResultList.append(mcResults)
        
        return FgPCResultList, mcResultList
    
    def getStochastics(self, 
                       myFgPCmodel: FgPC, 
                       FgPCResultList: list,
                       mcResultList: list, 
                       nrEvalPts: int, 
                       nrVar: int, 
                       samplingTimeIdxList: list,
                       confidence: float = 0.95,
                       forced: bool = False):
        r"""
        Method to calculate the stochastics of the FgPC and MC models

        Parameters
        ----------
        myFgPCmodel : FgPC object
            instance of the FgPC model
        FgPCResultList : list
            list with the results of the FgPC model
        mcResultList : list
            list with the results of the Monte Carlo model
        nrEvalPts : int
            number of evaluation points over one period
        nrVar : int
            number of variables in the system
        samplingTimeIdxList : list
            list with time points for which the marginal distribution 
            should be calculated
        confidence : float (default = 0.95)
            gives percentage of samples containing in the sampling interval
        forces : bool (default = False)
            boolean to indicate if the system is forced
        
        Returns
        ----------
        The stochastics contains the mean, and the coverage boundaries 
        of the corresponding position and velocity of the systems variables.

        FgPCStochasticList : list
            list with the stochastics of the FgPC model
        mcStochasticList : list
            list with the stochastics of the Monte Carlo model
        diffStochasticList : list
            list with the difference of the stochastics between the FgPC and MC model
        FgPCsamplingList : list
            list with the sampling of the FgPC model at the given time points
        mcsamplingList : list
            list with the sampling of the MC model at the given time points
        """

        FgPCStochasticList = []
        mcStochasticList = []
        diffStochasticList = []
        FgPCsamplingList = []
        mcsamplingList = []
        for FgPCResults, mcResults in zip(FgPCResultList,mcResultList):

            posFgPC = np.zeros((FgPCResults.shape[0],nrEvalPts*nrVar))
            velFgPC = np.zeros((FgPCResults.shape[0],nrEvalPts*nrVar))
            posmc = np.zeros((mcResults.shape[0],nrEvalPts*nrVar))
            velmc = np.zeros((mcResults.shape[0],nrEvalPts*nrVar))
            
            if not forced:
                omegaFgPC = np.zeros((mcResults.shape[0],))
                omegamc = np.zeros((mcResults.shape[0],))
                for i in range(FgPCResults.shape[0]):
                    FgPCCoeffs = FgPCResults[i,:]
                    mcCoeffs = mcResults[i,:]
                    posFgPCCur, velFgPCCur, omegaFgPCCur = \
                        self.getDynamicSolutions(myFgPCmodel,FgPCCoeffs,forced)
                    posFgPC[i,:] = posFgPCCur
                    velFgPC[i,:] = velFgPCCur
                    omegaFgPC[i] = omegaFgPCCur
                    posmcCur, velmcCur, omegamcCur = \
                        self.getDynamicSolutions(myFgPCmodel,mcCoeffs,forced)
                    posmc[i,:] = posmcCur
                    velmc[i,:] = velmcCur
                    omegamc[i] = omegamcCur

            else:
                for i in range(FgPCResults.shape[0]):
                    FgPCCoeffs = FgPCResults[i,:]
                    mcCoeffs = mcResults[i,:]
                    posFgPCCur, velFgPCCur = \
                        self.getDynamicSolutions(myFgPCmodel,FgPCCoeffs)
                    posFgPC[i,:] = posFgPCCur
                    velFgPC[i,:] = velFgPCCur
                    posmcCur, velmcCur = \
                        self.getDynamicSolutions(myFgPCmodel,mcCoeffs)
                    posmc[i,:] = posmcCur
                    velmc[i,:] = velmcCur

            if self.logger is not None:
                self.logger.info("Stochastics: Calculated the Dynamics")
            
            # --- Stochastics ---
            FgPCStochastics = []
            mcStochastics = []
            diffStochastics = []
            FgPCSolSampling = []
            mcSolSampling = []

            for i in range(nrVar):

                FgPCSolVarSampling = []
                mcSolVarSampling = []
                for idxTime in samplingTimeIdxList:
                    FgPCSolVarSampling.append(posFgPC[:,nrEvalPts*i+idxTime])
                    mcSolVarSampling.append(posmc[:,nrEvalPts*i+idxTime])
                FgPCSolSampling.append(FgPCSolVarSampling)
                mcSolSampling.append(mcSolVarSampling)
            
            
                posMeanFgPC = np.zeros((nrEvalPts,))
                velMeanFgPC = np.zeros((nrEvalPts,))
                posConfLowFgPC = np.zeros((nrEvalPts,))
                posConfUpFgPC = np.zeros((nrEvalPts,))
                velConfLowFgPC = np.zeros((nrEvalPts,))
                velConfUpFgPC = np.zeros((nrEvalPts,))
                posMeanMC = np.zeros((nrEvalPts,))
                velMeanMC = np.zeros((nrEvalPts,))
                posConfLowMC = np.zeros((nrEvalPts,))
                posConfUpMC = np.zeros((nrEvalPts,))
                velConfLowMC = np.zeros((nrEvalPts,))
                velConfUpMC = np.zeros((nrEvalPts,))
                
                for j in range(nrEvalPts):
                    curposFgPC = posFgPC[:,nrEvalPts*i+j]
                    curvelFgPC = velFgPC[:,nrEvalPts*i+j]

                    curposmc = posmc[:,nrEvalPts*i+j]
                    curvelmc = velmc[:,nrEvalPts*i+j]

                    posMeanFgPC[j], velMeanFgPC[j], \
                        posConfLowFgPC[j], posConfUpFgPC[j], \
                            velConfLowFgPC[j], velConfUpFgPC[j] = \
                                self.getMeanConfidencePosVel(curposFgPC,curvelFgPC,
                                                             confidence=confidence)
                
                    posMeanMC[j], velMeanMC[j], \
                        posConfLowMC[j], posConfUpMC[j], \
                            velConfLowMC[j], velConfUpMC[j] = \
                                self.getMeanConfidencePosVel(curposmc,curvelmc,
                                                             confidence=confidence)

                FgPCVal = [posMeanFgPC,velMeanFgPC,
                            posConfLowFgPC,posConfUpFgPC,
                            velConfLowFgPC,velConfUpFgPC]
                FgPCStochastics.append(FgPCVal)
                
                mcVal = [posMeanMC,velMeanMC,
                         posConfLowMC,posConfUpMC,
                         velConfLowMC,velConfUpMC]
                mcStochastics.append(mcVal)
                
                curdiffStochastics = []
                for valMC, valFgPC in zip(mcVal, FgPCVal):
                    diffVal = np.abs(valMC - valFgPC)
                    curdiffStochastics.append(diffVal)
                diffStochastics.append(curdiffStochastics)
                
            if not forced:
                omegaMean, _, omegaConfLow, omegaConfUp, _, _ = \
                    self.getMeanConfidencePosVel(omegaFgPC, confidence=confidence)
                omegaFgPCStochastics = [omegaFgPC, omegaMean, 
                                         omegaConfLow,omegaConfUp]
                FgPCSolSampling.append(omegaFgPCStochastics)
                mcSolSampling.append(omegamc)

            FgPCStochasticList.append(FgPCStochastics)
            mcStochasticList.append(mcStochastics)
            diffStochasticList.append(diffStochastics)
            FgPCsamplingList.append(FgPCSolSampling)
            mcsamplingList.append(mcSolSampling)
            
            if self.logger is not None:
                self.logger.info("Stochastics: Calculated the stochastics")
        
        return FgPCStochasticList, mcStochasticList, \
            diffStochasticList, FgPCsamplingList, mcsamplingList

    def getDynamicSolutions(self, 
                            myFgPCmodel: FgPC,
                            hbCoeffs: np.ndarray, 
                            forced: bool = True):
        r"""
        Method to calculate the dynamic solutions of the system.

        Parameters
        ----------
        myFgPCmodel : FgPC object
            instance of the FgPC model
        hbCoeffs : np.ndarray
            array with the HB coefficients
        forced : bool (default = True)
            boolean to indicate if the system is forced

        Returns
        ----------
        posCur : np.ndarray
            current position over one period
        velCur : np.ndarray
            current velocity over one period
        when not forced:
        omegaCur : float
            current base frequency
        """

        if not forced:
            fgpcCoeffCur, omegaCur = myFgPCmodel.splitCoeffVec(hbCoeffs)
            posCur, velCur = myFgPCmodel.getPosVel(fgpcCoeffCur, omegaCur)
            return posCur, velCur, omegaCur
        else:
            hbCoeffCompl = myFgPCmodel.convertAllHBSinCos2Complx(hbCoeffs)
            posCur = myFgPCmodel.calcPosition(hbCoeffCompl, myFgPCmodel.E_nh_c_total)
            velCur = myFgPCmodel.calcVelocity(hbCoeffCompl, myFgPCmodel.E_nh_c_total,
                                                myFgPCmodel.omega, myFgPCmodel.derMat_c_total)
            return posCur, velCur

    def getMeanConfidencePosVel(self, 
                                pos: np.ndarray,
                                vel: np.ndarray = None,
                                confidence: float = 0.95):
        r"""
        Method to calculate the mean and confidence interval of
        the position and velocity.

        Parameters
        ----------
        pos : np.ndarray
            position over one period
        vel : np.ndarray (default = None)
            velocity over one period
        confidence : float (default = 0.95)
            gives percentage of samples containing in the sampling interval
        
        Returns
        ----------
        meanPos : np.ndarray
            Mean position.
        meanVel : np.ndarray
            Mean velocity.
        confPosLow : np.ndarray
            Lower bound of confidence interval of the position.
        confVelLow : np.ndarray
            Lower bound of confidence interval of the velocity.
        confPosUp : np.ndarray
            Upper bound of confidence interval of the position.
        confVelUp : np.ndarray
            Upper bound of confidence interval of the velocity.
        """

        posMean = np.mean(pos, axis=0)
        
        n = len(pos)
        count = int(round(((1-confidence)/2)*n))
        posConfLow = pos[count-1]
        posConfUp = pos[-count]
        
        if vel is not None:
            velMean = np.mean(vel, axis=0)
            
            velConfLow = vel[count-1]
            velConfUp = vel[-count]
        else:
            velMean = None
            velConfLow = None
            velConfUp = None

        return posMean, velMean, posConfLow, posConfUp, velConfLow, velConfUp
    