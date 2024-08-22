r"""
Author:         Lars de Jong
Date:           2024-07-23
Description:    Class to calculate all convergence related
                information for the FgPC model.

"""
import os
import pickle
import numpy as np
import chaospy as cp

from fgPC.models.fgpc import fouriergenPolynomialChaos as FgPC

class convergenceFgPC:

    def __init__(self,
                 solutionMatrix: list,
                 varNr: int,
                 distStr: str,
                 low: float,
                 high: float,
                 sampleStr: str,
                 sampleNr: int,
                 forced: bool = True,
                 logger = None):
        
        self.solutionMatrix = solutionMatrix
        self.h = len(solutionMatrix)
        self.ngPC = len(solutionMatrix[0])
        self.varNr = varNr
        self.forced = forced

        self.distStr = distStr
        self.getDistribution(self.distStr, 
                             low, 
                             high)

        self.sampleNr = sampleNr
        self.sampleStr = sampleStr
        self.generatSamples()
        self.logger = logger

        if self.logger is not None:
            self.logger.info("Convergence model initialized.")

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

    def calcConvergenceMap(self,
                           myFgPCmodel: FgPC):

        # get reference solution
        for ih in range(self.h-1, 0, -1):
            for ingPC in range(self.ngPC-1, 0, -1):
                curSol = self.solutionMatrix[ih][ingPC]
                if curSol is not None:
                    if isinstance(curSol[0], list):
                        refCoeffVec = curSol[0][0]
                    else:
                        refCoeffVec = curSol[0]
                    refH = ih+1
                    refNgPC = ingPC+1
                    refSolFound = True
                    break
            if refSolFound:
                break

        myFgPCmodel.harmonics = [refH] * self.varNr
        myFgPCmodel.ngPC = refNgPC

        self.refIntSolMat = self.calcSampledTimeSolution(refCoeffVec,
                                                      myFgPCmodel)

        if self.logger is not None:
            self.logger.info("Reference solution created.")

        errorMat = np.zeros((self.h, self.ngPC))
        errorTensor = np.zeros((self.varNr, self.h, self.ngPC))
        # calculate error
        for jh in range(self.h):
            if jh > refH:
                break
            for jngPC in range(self.ngPC):
                if jngPC > refNgPC:
                    break

                curCoeffVec = self.solutionMatrix[jh][jngPC]

                if curCoeffVec is not None:
                    if isinstance(curCoeffVec[0], list):
                        curCoeffVec = curCoeffVec[0][0]
                    else:
                        curCoeffVec = curCoeffVec[0]

                    myFgPCmodel.harmonics = [jh+1] * self.varNr
                    myFgPCmodel.ngPC = jngPC+1

                    curIntSolMat = self.calcSampledTimeSolution(curCoeffVec,
                                                             myFgPCmodel)
                    
                    diffMat = (self.refIntSolMat -curIntSolMat)**2
                    
                    errorMat[jh,jngPC] = np.sum(np.sqrt(1/diffMat.shape[1] * 
                                                           np.sum(diffMat, axis=1)))
                    errorTensor[:,jh,jngPC] = np.sqrt(1/diffMat.shape[1] * np.sum(diffMat, axis=1))
                else:
                    errorMat[jh,jngPC] = np.nan
                    errorTensor[:,jh,jngPC] = np.nan
        
        if self.logger is not None:
            self.logger.info("Convergence tensor calculated.")
        return errorTensor, errorMat

    def calcSampledTimeSolution(self,
                                coeffVec: np.ndarray,
                                myFgPCmodel: FgPC):
        r"""
        Method to calculate the integral of over time of all sample solutions

        Parameters
        ----------
        coeffVec : np.ndarray
            vector with the coefficients of the solution
        myFgPCmodel : FgPC object
            instance of the FgPC model
        
        Returns
        ----------
        timeIntSolMat : np.ndarray
            matrix with the integral solutions for each variable (axis 0)
            of all samples (axis 1)
        """

        # load samples
        with open(self.sampleStr, "rb") as f:
            samples = pickle.load(f)

        timeIntSolMat = np.zeros((self.varNr,samples.shape[0]))
        for sample, i in zip(samples, range(samples.shape[0])):

            if not self.forced:
                fgpcCoeffs, _ = myFgPCmodel.constructCosSinFgPCCoeffs(coeffVec)
            else:
                fgpcCoeffs = coeffVec

            ident = np.eye(myFgPCmodel.totalH)
            polyVals = myFgPCmodel.calculatePolynomials(sample)
            hbCoeffs = np.dot(np.kron(ident,polyVals), fgpcCoeffs)

            hbCoeffsCompl = myFgPCmodel.convertAllHBSinCos2Complx(hbCoeffs)
            pos = myFgPCmodel.calcPosition(hbCoeffsCompl, myFgPCmodel.E_nh_c_total)

            for j in range(self.varNr):
                posVar = pos[j*myFgPCmodel.nrEvalPts:(j+1)*myFgPCmodel.nrEvalPts]
                timeIntSolMat[j,i] = 1/posVar.shape[0] * np.sum(np.abs(posVar),axis=0)

        return timeIntSolMat
    
    def calcRMSE(self,
                 curIntSolMat: np.ndarray,
                 nrEvalPts: int):
        r"""
        Method to calculate the RMSE between the current and reference solutions

        Parameters
        ----------
        curIntSolMat : np.ndarray
            matrix with the integral solutions for each variable (axis 0)
            of all samples (axis 1)
        nrEvalPts : int
            number of evaluation points over one period
        
        Returns
        ----------
        errorCoeff : np.ndarray
            RMSE of the current solutions compared to reference solution
            for each variable
        """

        diffMat = (self.refIntSolMat -curIntSolMat)**2

        error = np.sqrt(1/diffMat.shape[1] * np.sum(diffMat, axis=1))

        return error
        