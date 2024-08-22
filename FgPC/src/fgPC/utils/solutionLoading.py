r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    Loads previous calculated solutions of the FgPC model.

"""
import os
import re
import pickle
import argparse
import numpy as np


def loadSolution(desH: int,
                 desgPC: int, 
                 disStr: str,
                 solTag: str, 
                 varNr: int = 1, 
                 forced: bool = False):
    r""" Method to load previous calculated solutions of the FgPC model.

    Parameters
    ----------
    desH : int
        desired number of harmonics
    desgPC : int
        desired degree of the gPC
    disStr : str
        distribution of the system
    solTag : str
        solution tag
    varNr : int
        number of variables in the system
    forced : bool
        boolean to indicate if the system is forced

    Returns
    ----------
    FgPCSolList : list
        list with all solutions corresponding to of the FgPC model
    """

    fodlerStr = "0_solutionData/"
    
    matchingsolutions = [file for file in os.listdir(fodlerStr) if disStr in file and solTag in file]
    matchingsolutions.reverse()
    
    # Regular expression pattern to extract numbers following 'A' and 'B'
    pattern = r'(H\d+)_.*?(gPC\d+)_'

    solutionList = []
    # Extract numbers from the filenames
    for file_name in matchingsolutions:
        matches = re.findall(pattern, file_name)

        h = int(matches[0][0][1:])
        gPC = int(matches[0][1][3:])

        with open(fodlerStr + file_name, "rb") as f:
            solutionLoad = pickle.load(f)
        
        for sols in solutionLoad:
            solutionList.append((h, gPC, sols))

    # create initial guess from the solution
    FgPCSolList = []
    for solution in solutionList:
        h = solution[0]
        gPC = solution[1]
        if not forced:
            sol = solution[2][0]
            ampSol = solution[2][1]
        else:
            sol = solution[2]

        if len(sol)!=(2*h+1)*gPC*varNr:
            raise ValueError('Solution does not match the given configuration  \n Method for qt = 0 not implemented')
        
        
        if h == desH and gPC == desgPC:
            if not forced:
                FgPCSolList.append([sol, ampSol])
            else:
                FgPCSolList.append(sol)
        else:

            if not forced:
                newInitGuess = initialGuessSelfexcited(desH, desgPC, sol, varNr, gPC, h)
                FgPCSolList.append([newInitGuess, sol[1]])
            else:
                newInitGuess = initialGuessForced(desH, desgPC, sol, varNr, gPC, h)
                FgPCSolList.append(newInitGuess)

    return FgPCSolList


def initialGuessSelfexcited(desH: int,
                            desgPC: int, 
                            sol: np.ndarray, 
                            varNr: int, 
                            gPC: int, 
                            h: int):
    r"""
    Method to create an initial guess for the FgPC model of 
    the self-excited system by using a previous calculated solution.

    Parameters
    ----------
    desH : int
        desired number of harmonics
    desgPC : int
        desired degree of the gPC
    sol : np.array
        array of a calculated solution
    varNr : int
        number of variables in the system
    gPC : int
        desired degree of the gPC
    h : int
        desired number of harmonics

    Returns
    ----------
    newInitGuess : np.array
        array with the new initial guess
    """
    
    # Extract the omega coefficients
    newOmegaCoeff = np.zeros(desgPC)
    omegaCoeff = sol[-gPC:]
    if gPC < desgPC:
        hOmega = np.zeros(desgPC)*omegaCoeff[-1]
        hOmega[:gPC] = omegaCoeff
        newOmegaCoeff = hOmega
    else:
        newOmegaCoeff = omegaCoeff[:desgPC]

    # Extract cos and sin coefficients depending on h and gPc
    varList = []
    gesH = 0
    for idxVar in range(varNr):
        newconsCoeff = np.zeros(desgPC)
        newcosCoeff = np.zeros(desH*desgPC)
        if idxVar == 0:
            newsinCoeff = np.zeros((desH-1)*desgPC)
            var = sol[:2*h*gPC]
            varCons = var[:gPC]
            varCos = var[gPC:(h+1)*gPC]
            varSin = var[(h+1)*gPC:]
            iBase = 2*h*gPC
        else:
            newsinCoeff = np.zeros(desH*desgPC)
            var = sol[gesH+iBase:gesH+iBase+(2*h+1)*gPC]
            varCons = var[:gPC]
            varCos = var[gPC:(h+1)*gPC]
            varSin = var[(h+1)*gPC:]
            gesH += (2*h+1)*gPC

        sinIdx = 0
        for idxH in range(0,desH):
            if idxH == 0:
                if gPC < desgPC:
                    hCons = np.zeros(desgPC)*varCons[-1]
                    hCons[:gPC] = varCons
                    newconsCoeff = hCons

                    hCos = np.zeros(desgPC)*varCos[gPC-1]
                    hCos[:gPC] = varCos[:gPC]
                    newcosCoeff[:desgPC] = hCos
                    if idxVar != 0:
                        hSin = np.zeros(desgPC)*varSin[gPC-1]
                        hSin[:gPC] = varSin[:gPC]
                        newsinCoeff[:desgPC] = hSin
                        sinIdx = 1
                else:
                    newconsCoeff = varCons[:desgPC]
                    newcosCoeff[:desgPC] = varCos[:desgPC]
                    if idxVar != 0:
                        newsinCoeff[:desgPC] = varSin[:desgPC]
                        sinIdx = 1
            else:
                if idxH < h:
                    if gPC < desgPC:
                        hCos = np.zeros(desgPC)*varCos[(idxH+1)*gPC-1]
                        hCos[:gPC] = varCos[idxH*gPC:(idxH+1)*gPC]

                        hSin = np.zeros(desgPC)*varSin[(idxH+sinIdx)*gPC-1]
                        hSin[:gPC] = varSin[(idxH-1+sinIdx)*gPC:(idxH+sinIdx)*gPC]

                        newcosCoeff[idxH*desgPC:(idxH+1)*desgPC] = hCos
                        newsinCoeff[(idxH-1+sinIdx)*desgPC:(idxH+sinIdx)*desgPC] = hSin
                    else:
                        newcosCoeff[idxH*desgPC:(idxH+1)*desgPC] = varCos[idxH*gPC:idxH*gPC+desgPC]
                        newsinCoeff[(idxH-1+sinIdx)*desgPC:(idxH+sinIdx)*desgPC] = varSin[(idxH-1+sinIdx)*gPC:(idxH-1+sinIdx)*gPC+desgPC]
                else:
                    newcosCoeff[idxH*desgPC:(idxH+1)*desgPC] = np.zeros(desgPC)
                    newsinCoeff[(idxH-1+sinIdx)*desgPC:(idxH+sinIdx)*desgPC] = np.zeros(desgPC)

        varList.append(np.concatenate((newconsCoeff, newcosCoeff, newsinCoeff)))
    
    newInitGuess = np.concatenate((np.concatenate((varList)), newOmegaCoeff))

    return newInitGuess

def initialGuessForced(desH: int,
                       desgPC: int, 
                       sol: np.ndarray, 
                       varNr: int, 
                       gPC: int, 
                       h:int):
    r"""
    Method to create an initial guess for the FgPC model of
    the forced system by using a previous calculated solution.

    Parameters
    ----------
    desH : int
        desired number of harmonics
    desgPC : int
        desired degree of the gPC
    sol : np.ndarray
        array of a calculated solution
    varNr : int
        number of variables in the system
    gPC : int
        desired degree of the gPC
    h : int
        desired number of harmonics
    
    Returns
    ----------
    newInitGuess : np.ndarray
        array with the new initial guess
    """
    
    varList = []
    gesH = 0
    for idxVar in range(varNr):
        newcosCoeff = np.zeros(desH*desgPC)
        newsinCoeff = np.zeros(desH*desgPC)
        var = sol[gesH:gesH+(2*h+1)*gPC]
        varCons = var[:gPC]
        if gPC < desgPC:
            cons = np.zeros(desgPC)*varCons[-1]
            cons[:gPC] = varCons
            newconsCoeff = cons
        else:
            newconsCoeff = varCons[:desgPC]
        
        varCos = var[gPC:(h+1)*gPC]
        varSin = var[(h+1)*gPC:]
        gesH += (2*h+1)*gPC

        for idxH in range(0,desH):
            if idxH < h:
                if gPC < desgPC:
                    hCos = np.zeros(desgPC)*varCos[(idxH+1)*gPC-1]
                    hCos[:gPC] = varCos[idxH*gPC:(idxH+1)*gPC]

                    hSin = np.zeros(desgPC)*varSin[(idxH+1)*gPC-1]
                    hSin[:gPC] = varSin[idxH*gPC:(idxH+1)*gPC]

                    newcosCoeff[idxH*desgPC:(idxH+1)*desgPC] = hCos
                    newsinCoeff[idxH*desgPC:(idxH+1)*desgPC] = hSin
                else:
                    newcosCoeff[idxH*desgPC:(idxH+1)*desgPC] = varCos[idxH*gPC:idxH*gPC+desgPC]
                    newsinCoeff[idxH*desgPC:(idxH+1)*desgPC] = varSin[idxH*gPC:idxH*gPC+desgPC]
            else:
                newcosCoeff[idxH*desgPC:(idxH+1)*desgPC] = np.zeros(desgPC)
                newsinCoeff[idxH*desgPC:(idxH+1)*desgPC] = np.zeros(desgPC)

        varList.append(np.concatenate((newconsCoeff, newcosCoeff, newsinCoeff)))
    
    newInitGuess = np.concatenate((varList))

    return newInitGuess

                

