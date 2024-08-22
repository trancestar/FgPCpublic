r"""
Module: FgPC.py
Author: Lars de Jong
Date: 2024-02-10
Description: This file contains the super class for all harmonic balance
             generalized Polynomial Chaos (HB-GPC) models.
"""
import argparse
import logging
import time

import numpy as np
import chaospy as cp

from scipy.optimize import root, fsolve
from abc import ABC, abstractmethod


class fouriergenPolynomialChaos(ABC):
    r"""
    This is the super class for all harmonic balance generalized Polynomial
    Chaos (HB-GPC) models.
    """
    def __init__(self, 
                 harmonics: list, 
                 nrEvalPts: int, 
                 ngPC: int, 
                 nrQuadPts: int,
                 dist: str, 
                 low: float, 
                 high: float,
                 amp_s1: float = 0,
                 config: argparse.Namespace = None, 
                 logger: logging.Logger = None,
                 deflation: bool = False, **kwargs):
        r"""
        Constructor for the harmonicBalancegenPolynomialChaos class.

        Parameters
        ----------
        harmonics : list
            List of the harmonics.
        nrEvalPts : int
            Number of evaluation points within one period.
        ngPC : int
            degree of polynomials.
        nrQuadPts : int
            Number of quadrature points.
        dist : str
            Distribution of the uncertain system.
        low : float
            either lower bound or mean of the distribution
        high : float
            either upper bound or standard deviation of the distribution
        amp_s1 : float (default = 0)
            Sine amplitude of the first harmonic.
        config : argparse.Namespace (default = None)
            Configuration parameters of the FgPC model.
        logger : logging.Logger (default = None)
            Logger object.
        deflation : bool (default = False)
            Boolean to indicate if deflation is used.
        **kwargs : dict
            Additional key word arguments.
            quadRule : str
                Rule for quadrature. For options see chaospy generate_quadrature.
            tol : float
                Tolerance for root finding.
        """
        self.config = config
        self.logger = logger

        self._harmonics = harmonics
        self._updateTotalH()
        self.nrEvalPts = nrEvalPts

        self._ngPC = ngPC
        self.nrQuadPts = nrQuadPts

        self.distStr = dist
        self.getDistribution(dist, low, high)

        if "quadRule" in kwargs:
            self.quadRule = kwargs["quadRule"]
        else:
            self.quadRule = "gaussian"

        self.getQuadratureValues()
        self.getPolynomials()

        if "tol" in kwargs:
            self.tol = kwargs["tol"]
        else:
            self.tol = None
        
        self.calculateFourierTransformers()

        self.amp_s1 = amp_s1

        self.itrCt = 0

        self.deflation = deflation
        self.solutionList = []
        self.solCt = 0
        self.p = 3
        self.alpha_def = 1

        if self.logger is not None:
            self.logger.info("FgPC model initialized.")

    @property
    def harmonics(self):
        return self._harmonics
    
    @harmonics.setter
    def harmonics(self, value):
        self._harmonics = value
        self._updateTotalH()
        self.calculateFourierTransformers()

    @property
    def totalH(self):
        return self._totalH
    
    @property
    def ngPC(self):
        return self._ngPC
    
    @ngPC.setter
    def ngPC(self, value):
        self._ngPC = value
        self.getQuadratureValues()
        self.getPolynomials()
    
    def _updateTotalH(self):
        self._totalH = np.sum(np.array([self.harmonics])*2+1)

    @abstractmethod
    def constructCosSinFgPCCoeffs(self):
        r"""
        Abstract method to construct the FgPC coefficients when they given
        in cosine and sine order.
        """
        pass

    @abstractmethod 
    def calculateFgPC(self):
        r"""
        Abstract method to calculate the HB-GPC model.
        """
        pass

    def calculategPCinnerProduct(self, 
                                 currentGuess: np.ndarray):
        r"""
        Method that calculates the inner product of the FgPC model.

        Parameters
        ----------
        currentGuess : np.ndarray
            The current guess for the HB-GPC model. First the 
            constant term, then the cosines and then the sines.
        
        Returns
        ----------
        innerProduct : np.ndarray
            The inner product of the gPC coefficients.
        """

        t0_total = time.time()
        innerProduct = np.zeros(currentGuess.shape)

        i = 0
        for quadPtVal in self.quadPts[0]:
            quadPt = np.array([quadPtVal])
            
            polyVals = self.calculatePolynomials(quadPt)
            
            timeResList = self.calculateFgPC(currentGuess,
                                              quadPt,
                                              polyVals)
       
            freqResiduum = self.calcFreqResiduum(timeResList)
            
            integrands = self.calculateIntegrandIP(freqResiduum, polyVals)
            
            inPrcPt = integrands* self.quadWts[i]
            i += 1
            innerProduct += inPrcPt

        if self.deflation:
            deflationOperator = self.getDeflationOperator(currentGuess)
            innerProduct = np.dot(deflationOperator, innerProduct)

        t1_total = time.time()
        self.itrCt += 1
        print("Iteration: " + str(self.itrCt) + 
              " Function Eval Time: " + '{:.8f}'.format(t1_total-t0_total) +
              " Residuum Norm: " + '{:.6e}'.format(np.sqrt(np.sum(innerProduct**2))), end='\r')
        return innerProduct

    def calculateFourierTransformers(self):
        r"""
        Method calculates the Fourier Transformers according to Krack et al. 2019
        """
        
        evalPts = np.arange(0,self.nrEvalPts)
        E_nhList = []
        E_hnsList = []
        derMatList = []
        E_nh_total = np.zeros((self.nrEvalPts*len(self.harmonics), self.totalH), dtype=np.complex_)
        E_hns_total = np.zeros((self.totalH, self.nrEvalPts*len(self.harmonics)), dtype=np.complex_)
        derMat_total = np.zeros((self.totalH, self.totalH), dtype=np.complex_)

        rowInd = 0
        colInd = 0
        for varH in self.harmonics:
            harmonics = np.arange(-varH,varH+1)

            E_nh = np.dot(evalPts[:,np.newaxis],harmonics[np.newaxis,:])
            E_nh = np.exp(1j * 2* np.pi* E_nh / self.nrEvalPts)
            E_hns = np.dot(harmonics[:,np.newaxis],evalPts[np.newaxis,:])
            E_hns = np.exp(-1j * 2* np.pi* E_hns / self.nrEvalPts) / self.nrEvalPts
            
            derMat = np.diag(harmonics)*1j

            E_nhList.append(E_nh)
            E_hnsList.append(E_hns)
            derMatList.append(derMat)

            sizeRow = E_nh.shape[0]
            sizeCol = E_nh.shape[1]
            E_nh_total[rowInd:rowInd+sizeRow, colInd:colInd+sizeCol] = E_nh
            E_hns_total[colInd:colInd+sizeCol, rowInd:rowInd+sizeRow] = E_hns
            derMat_total[colInd:colInd+sizeCol, colInd:colInd+sizeCol] = derMat
            rowInd += sizeRow
            colInd += sizeCol
        
        self.E_nh_c_List = E_nhList
        self.E_hns_c_List = E_hnsList
        self.derMat_c_List = derMatList
        self.E_nh_c_total = E_nh_total
        self.E_hns_c_total = E_hns_total
        self.derMat_c_total = derMat_total

        if self.logger is not None: 
            self.logger.info("Calculated Fourier transformers \
                              and derivative matrix.")

    def convertAllHBSinCos2Complx(self, 
                                  fourierCoeff: np.ndarray):
        r"""
        Method converts the Fourier coefficients from sin/cos to complex

        Parameters
        ----------
        fourierCoeff : np.ndarray
            Fourier coefficients sorted in the following order:
            constant, cosines, sines
        
        Returns
        ----------
        complFC : np.ndarray
            complex Fourier coefficients
        """ 
        
        complFC_total = np.zeros(fourierCoeff.shape, dtype=np.complex_)
        indh = 0
        for h in self.harmonics:
            hNr = 2*h+1
            x_const = fourierCoeff[indh]
            x_c = fourierCoeff[indh+1:indh+h+1]
            x_s = fourierCoeff[indh+h+1:indh+2*h+1]

            complFC = np.zeros(hNr, dtype=np.complex_)

            complFC[h,] = x_const
            complFC[:h,] = 0.5 * (np.flip(x_c, axis=0) + 1j * np.flip(x_s, axis=0))
            complFC[h+1:,] = 0.5 * (x_c - 1j * x_s)

            complFC_total[indh:indh+hNr,] = complFC
            indh += hNr
        
        return complFC_total
    
    def performInverseFourierTransform4Vec(self, 
                                           resTime: np.ndarray, 
                                           E_hns_c: np.ndarray):
        r"""
        Method performs the inverse Fourier transform

        Parameters
        ----------
        resTime : np.ndarray
            residuum in time domain
        E_hns_c : np.ndarray
            Fast Fourier Transform matrix
        
        Returns
        ----------
        residuum : np.ndarray
            Residuum in in frequeny domain
        """ 

        residuumFC = np.dot(E_hns_c, resTime)

        residuumVec = np.array([])
        indH = 0
        for h in self.harmonics:
            hT = 2*h+1
            residuum = np.concatenate((
                np.asarray([np.real(residuumFC[indH+h,])]),
                np.real(residuumFC[indH+h+1:indH+2*h+1,])*2,
                -np.imag(residuumFC[indH+h+1:indH+2*h+1,])*2
            ))
            indH += hT
            residuumVec = np.append(residuumVec,residuum)
        
        return residuumVec
    

    def calcFreqResiduum(self, 
                         timeRes: np.ndarray):
        r"""
        Method to calculate the residuum in frequency domain.
        
        Parameters
        ----------
        timeResList : np.ndarray
            List of the residuum in time domain.
        
        Returns
        ----------
        freqResList : np.ndarray
            Array of the residuum in frequency domain.
        """
        
        freqRes = self.performInverseFourierTransform4Vec(timeRes,self.E_hns_c_total)

        return freqRes
    
    def calculateIntegrandIP(self, 
                             freqResiduum: np.ndarray, 
                             polyVals: np.ndarray):
        r"""
        Method to calculate the integrand for the inner product.
        
        Parameters
        ----------
        freqResiduum : np.ndarray
            residuum in frequency domain.
        polyVals : np.ndarray
            Polynomials evaluated at quadrature points
        
        Returns
        ----------
        integrands : np.ndarray
            solution of the second inner product.
        """
        
        integrands = np.kron(freqResiduum,polyVals).flatten()
        return integrands

    def calcPosition(self, 
                     complFC: np.ndarray, 
                     E_nh_c: np.ndarray):
        r"""
        Method calculates the position from the Fourier coefficients

        Parameters
        ----------
        fourierCoeff : np.ndarray
            complex Fourier coefficients sorted in the following order:
            constant, cosines, sines. Repeated for each variable
        E_nh_c : np.ndarray
            Inverse Fourier Transform matrix
        
        Returns
        ----------
        x : np.ndarray
            position at evaluation points over one period
        """ 
        
        x = np.real(np.dot(E_nh_c, complFC))
        return x
    
    def calcVelocity(self, 
                     complFC: np.ndarray, 
                     E_nh_c: np.ndarray, 
                     omega: float, 
                     derMat_c: np.ndarray):
        r"""
        Method calculates the velocity from the Fourier coefficients

        Parameters
        ----------
        fourierCoeff : np.ndarray
            complex Fourier coefficients sorted in the following order:
            constant, cosines, sines. Repeated for each variable
        E_nh_c : np.ndarray
            Inverse Fourier Transform matrix
        omega : float
            base frequency of the oscillation
        derMat_c : np.ndarray
            diagonal matrix with harmonic integer and imaginary unit
        
        Returns
        ----------
        xdot : np.ndarray
            velocity at evaluation points over one period
        """ 
        
        xdot = np.real(np.dot(E_nh_c, omega * \
                              np.dot(derMat_c, complFC)))
        return xdot

    def calcAcceleration(self, 
                         complFC: np.ndarray, 
                         E_nh_c: np.ndarray, 
                         omega: float, 
                         derMat_c: np.ndarray):
        r"""
        Method calculates the velocity from the Fourier coefficients

        Parameters
        ----------
        fourierCoeff : np.ndarray
            complex Fourier coefficients sorted in the following order:
            constant, cosines, sines. Repeated for each variable
        E_nh_c : np.ndarray
            Inverse Fourier Transform matrix
        omega : float
            base frequency of the oscillation
        derMat_c : np.ndarray
            diagonal matrix with harmonic integer and imaginary unit
        
        Returns
        ----------
        xddot : np.ndarray
            acceleration at evaluation points over one period
        """ 
        
        xddot = np.real(np.dot(E_nh_c, omega**2 * \
                               np.dot(derMat_c**2, complFC)))
        return xddot

    def getDistribution(self, 
                        distStr: str, 
                        low: float, 
                        high: float):
        r"""
        Method calculates the distribution
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
            self.lowLimit = self.dist.lower
            self.highLimit = self.dist.upper
        elif distStr == "beta":
            self.lowLimit = low
            self.highLimit = high
            self.dist = cp.Beta(5,5,low, high)
        else:
            raise ValueError("Distribution not implemented.")
        self.pdf = self.dist.pdf

    def getQuadratureValues(self):
        r"""
        This method calculates the quadrature points and weights
        for the given distribution.
        """

        if self.nrQuadPts < self.ngPC*2+1:
            raise ValueError("Number of quadrature points too low.")
        
        self.quadPts, self.quadWts = cp.generate_quadrature(self.nrQuadPts, self.dist,
                                            rule=self.quadRule)

    def getPolynomials(self):
        r"""
        This method calculates the polynomials for the given distribution.
        """
        self.polynomials = cp.generate_expansion(self.ngPC-1, self.dist, normed=True)

    def calculatePolynomials(self, 
                             quadPt: np.ndarray):
        r"""
        This method returns an 1d array with the polynomial values
        evaluated at the given points
        
        Parameters
        ----------
        quadPt : np.ndarray
            Quadrature points where the polynomials should be evaluated.

        Returns
        ----------
        polyVals : np.ndarray
            The polynomial values at the given points.
        """

        gPC = np.arange(self.ngPC)
        polyVals = self.polynomials[gPC](quadPt)
        
        return polyVals.transpose()

    def getDeflationOperator(self,
                             x: np.ndarray):
        r"""
        Method returns the deflation operator for current guess

        Parameters
        ----------
        x : np.ndarray
            current guess of FgPC coefficients

        Returns
        ----------
        deflationOperator : np.ndarray
            deflation operator for the current guess
        """
        if len(self.solutionList) == 0:
            deflationOperator = np.identity(x.shape[0]) 
        else:
            if len(self.solutionList) > self.solCt:
                i = 0
                for sol in self.solutionList:
                    identityMat = np.identity(sol.shape[0])
                    normVal = np.linalg.norm(x-sol)
                    deflationOpera = identityMat/(normVal**self.p)+self.alpha_def*identityMat
                    if i==0:
                        deflationOperator = deflationOpera
                        i = 1
                    else:
                        deflationOperator = np.matmul(deflationOperator,deflationOpera)
                self.deflationOperator = deflationOperator
                self.solCt
            else:
                deflationOperator = self.deflationOperator

        return deflationOperator