r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    The genereal procedure to calculate the 
                solution of the FgPC model.

"""
import os
import argparse
import logging
import pickle

import numpy as np

from scipy.optimize import root

from fgPC.models.fgpc import fouriergenPolynomialChaos as FgPC
from fgPC.models.initialGuessFunctions import InitialGuessFunctions as InitGuess

from fgPC.utils.solutionLoading import loadSolution

class FgPCsolution:

    def __init__(self,
                 configFgPC: argparse.Namespace, 
                 configInt: argparse.Namespace,
                 resultFolder: str,
                 systemStr: str,
                 distributionStr: str,
                 tag: str,
                 varNr: int,
                 forced: bool,
                 exptSol: int = 1,
                 logger: logging.Logger = None,
                 deflationBool: bool = False):
        r"""
        Constructor of the FgPCsolution class

        Parameters
        ----------
        configFgPC : argparse.Namespace
            configuration parameters for the FgPC model
        configInt : argparse.Namespace
            configuration parameters for the integration
        resultFolder : str
            folder where the results are stored
        systemStr : str
            name of the system
        distributionStr : str
            distribution of the system
        tag : str
            solution tag
        varNr : int
            number of variables in the system
        forced : bool
            boolean to indicate if the system is forced
        exptSol : int (default = 1)
            number of expected solutions
        logger : Logger (default = None)
            instance of the logger
        deflationBool : bool (default = False)
            boolean to indicate if deflation is used
        """
        
        self.configFgPC = configFgPC
        self.configInt = configInt
        self.resultFolder = resultFolder
        self.systemStr = systemStr
        self.distributionStr = distributionStr
        self.tag = tag
        self.varNr = varNr
        self.forced = forced
        self.exptSol = exptSol
        self.logger = logger
        self.deflationBool = deflationBool

        if self.logger is not None:
            self.logger.info("solution class initialized")

    def calFgPCSolution(self,
                        FgPCmodel: FgPC,
                        initModel: InitGuess,
                        h: int,
                        ngPC: int):
        r"""
        Method to calculate the solution of the FgPC model

        Parameters
        ----------
        FgPCmodel : FgPC
            instance of the FgPC model
        initModel : InitGuess
            instance of the initial guess model
        h : int
            number of harmonics
        ngPC : int
            degree of polynomials
        
        Returns
        ----------
        FgPCmodel.solutionList : list
            list with the solutions of the FgPC model
        """

        solList = loadSolution(h,
                               ngPC,
                               self.distributionStr, 
                               self.tag, 
                               self.varNr, 
                               self.forced)
        
        initialGuessList = initModel.initialGuessFgPC(h,
                                                      ngPC,
                                                      self.configFgPC,
                                                      self.configInt,
                                                      self.resultFolder,
                                                      self.systemStr,
                                                      self.tag,
                                                      self.logger)
        solList = initialGuessList + solList
        
        if self.forced:
            initialGuess = solList[0]
        else:
            initialGuess = solList[0][0]
            FgPCmodel.amp_s1 = solList[0][1]

        solFound = 0
        i = 1
        solutionList = []
        FgPCmodel.itrCt = 0
        while solFound < self.exptSol:
        
            sol = root(FgPCmodel.calculategPCinnerProduct,
                       initialGuess,
                       tol = 1e-10,
                       options = {'maxfev':  1000*initialGuess.shape[0],
                                  'xtol': 1e-12})
            
            print('\n' + sol.message)
            if self.deflationBool:
                if sol.message == "The solution converged.":
                    if self.forced:
                        solutionList.append(sol.x)
                    else:
                        solutionList.append([sol.x, self.FgPCmodel.amp_s1])
                    solFound += 1
                    FgPCmodel.itrCt = 0

                    if self.logger is not None:
                        self.logger.info(sol.message + " For H = " + str(h) + " and gPC = " + str(ngPC))
                else:
                    if i < len(solList):
                        if self.forced:
                            initialGuess = solList[i]
                        else:
                            initialGuess = solList[i][0]
                            FgPCmodel.amp_s1 = solList[i][1]
                        i += 1
                    else:
                        raise ValueError("No all expected solution found")
                    
                    if self.logger is not None:
                        self.logger.info("No solution found for H = " + str(h) + " and gPC = " 
                                         + str(ngPC) + " Message: " + sol.message)
            else:
                if sol.message == "The solution converged.":
                    if self.forced:
                        solutionList.append(sol.x)
                    else:
                        solutionList.append([sol.x, FgPCmodel.amp_s1])
                    FgPCmodel.itrCt = 0
                    if self.logger is not None:
                        resNorm = np.sqrt(np.sum(FgPCmodel.calculategPCinnerProduct(sol.x)**2))
                        self.logger.info(sol.message + " For H = " + str(h) + " and gPC = " + str(ngPC) 
                                        + " with residual norm: " + '{:.6e}'.format(resNorm))
                    break
                else:
                    if len(solList) > 1 and i < len(solList) and FgPCmodel.itrCt < 1e5:
                        if self.forced:
                            initialGuess = solList[i]
                        else:
                            initialGuess = solList[i][0]
                            FgPCmodel.amp_s1 = solList[i][1]
                        i += 1
                    else:
                        if self.logger is not None:
                            self.logger.info("No solution with given initial guess found.")
                        raise ValueError("No solution found")
                    
                    if self.logger is not None:
                        self.logger.info("No solution found for H = " + str(h) + " and gPC = " 
                                         + str(ngPC) + " Message: " + sol.message)
                    
        return solutionList

    def calFgPCSolutionMatrix(self,
                              FgPCmodel: FgPC,
                              initModel: InitGuess,
                              solStrList: list):

        if self.deflationBool:

            solStr = solStrList[0] + "{:02}".format(self.configFgPC.H) \
                    + "_gPC" + "{:02}".format(self.configFgPC.ngPC) + solStrList[1]
            FgPCSolList = self.calFgPCSolution(FgPCmodel,
                                               initModel,
                                               self.configFgPC.H,
                                               self.configFgPC.ngPC)
            FgPCmodel.solutionList = FgPCSolList
            with open(solStr, "wb") as f:
                pickle.dump(FgPCSolList,f) 
            return FgPCSolList
        
        else:

            solMatStr1 = "1_allSolData/solMat_"+ self.tag + "_H" \
                + "{:02}".format(self.configFgPC.H) + "_gPC" \
                    + "{:02}".format(self.configFgPC.ngPC) + solStrList[1]
            
            if not os.path.exists(solMatStr1):
                solutionMatrix = [[None for _ in range(self.configFgPC.ngPC)] \
                                for _ in range(self.configFgPC.H)]
                for gPC in range(1,self.configFgPC.ngPC+1):
                    FgPCmodel.ngPC = gPC
                    for h in range(1,self.configFgPC.H+1):
                        FgPCmodel.harmonics = [h] * self.varNr

                        solStr = solStrList[0] + "{:02}".format(h) \
                            + "_gPC" + "{:02}".format(gPC) + solStrList[1]

                        if os.path.exists(solStr):
                            with open(solStr, "rb") as f:
                                curSol = pickle.load(f)
                            
                        else:
                            try:
                                curSol = self.calFgPCSolution(FgPCmodel,
                                                            initModel,
                                                            h,gPC)
                                
                                with open(solStr, "wb") as f:
                                    pickle.dump(curSol,f)
                            except ValueError:
                                curSol = None

                        solutionMatrix[h-1][gPC-1] = curSol
                
                with open(solMatStr1, "wb") as f:
                    pickle.dump(solutionMatrix,f)

            else:
                with open(solMatStr1, "rb") as f:
                    solutionMatrix = pickle.load(f)
                    
            return solutionMatrix

                    