r"""
Author:         Lars de Jong
Date:           2024-05-15
Description:    FgPC class of the beta cell system.

"""
import argparse
import logging
import time
import numpy as np

from fgPC.models.fgpc import fouriergenPolynomialChaos as FgPC
from cellBiologyFgPC.models.betaCellModel import betaCellfunctions

class betaCellFgPC(FgPC):
    r"""
    This is the class for the beta cell model using the harmonic balance
    generalized polynomial chaos (HB-GPC) method.
    """
    def __init__(self,
                 config: argparse.Namespace,
                 system_tag: str,
                 sinMt = 1,
                 amp_s1 = 0,
                 logger: logging.Logger = None,
                 **kwargs):
        r"""
        Constructor for the betaCellFgPC class.

        Parameters
        ----------
        config : argparse.Namespace
            The configuration object containg all input parameters.
        system_tag : str
            Tag defining the system to be used.
        sinMt : int (default = 1)
            Integer defining the sine mode to be used, either set
            to amp_s1 value (=1) or add extra condition dvdt(t=0) (=2).
        amp_s1 : float (default = 0)
            Amplitude of the first sine harmonics.
        logger : logging.Logger (default = None)
            The logger object to log messages.
        **kwargs : dict
            Additional key word arguments.
            quadRule : str
                rule for quadrature. For options see chaospy generate_quadrature.
            tol : float
                tolerance for root finding.
        """
        
        self.H = config.H

        self.system_tag = system_tag
        harmonics = [self.H] * 3
        
        super().__init__(harmonics, 
                         config.nPt, 
                         config.ngPC, 
                         config.nQPt,
                         config.distStr, 
                         config.low,
                         config.high, 
                         amp_s1,
                         config, 
                         logger,
                         **kwargs)

        self.betaCell = betaCellfunctions()
        self.sinMt = sinMt

        if self.logger is not None:
            self.logger.info("Beta cell FgPC model initialized.")

    def constructCosSinFgPCCoeffs(self, 
                                  currentGuessCosSin: np.ndarray):
        r"""
        Method to construct the FgPC coefficients when they given in cosine
        and sine order.

        Parameters
        ----------
        currentGuessCosSin : np.ndarray
            The current guess for the FgPC model. First the 
            gPC representation of the cosine and sine
            coefficients followed by the omega coefficients.

        Returns
        ----------
        FgPCCoeffs : np.ndarray
            The FgPC coefficients. The first sin coefficient added.
        omegaCoeffs : np.ndarray
            The omega gPC coefficients.
        """

        FgPCCoeffs = np.zeros(self.totalH*self.ngPC)
        h = self.harmonics[0]
        if self.sinMt == 1:
            sin1gPC = np.zeros(self.ngPC)
            sin1gPC[0] = self.amp_s1
            FgPCCoeffs[:(2*h+1)* self.ngPC] = np.concatenate([currentGuessCosSin[:(h+1)* self.ngPC],
                                    sin1gPC,
                                    currentGuessCosSin[(h+1)* self.ngPC:(2*h)* self.ngPC]])
        else:
            FgPCCoeffs[:(2*h+1)* self.ngPC] = currentGuessCosSin[:(2*h+1)* self.ngPC]
        
        gesIdx = (2*h+1)*self.ngPC
        FgPCCoeffs[gesIdx:] = currentGuessCosSin[gesIdx-self.ngPC:-self.ngPC]
        
        omegaCoeffs = currentGuessCosSin[-self.ngPC:]

        return FgPCCoeffs, omegaCoeffs
    
    def splitCoeffVec(self, 
                      currentGuessCosSin: np.ndarray):
        r"""
        Splits the current guess vector into the HB coefficients and the omega.
        Adds the first sine coefficient if sinMt is set to 1.
        ----------
        currentGuessCosSin : np.ndarray
            The current guess for the HB model.

        Returns
        ----------
        hbCoeffs : np.ndarray
            The HB coefficients.
        omega : float
            The omega value.
        """

        hbCoeffs = np.zeros(self.totalH)
        h = self.harmonics[0]
        if self.sinMt == 1:
            hbCoeffs[:(2*h+1)] = np.concatenate([currentGuessCosSin[:(h+1)],
                                    np.array([self.amp_s1]),
                                    currentGuessCosSin[(h+1):(2*h)]])
            gesIdx = (2*h)
        else:
            hbCoeffs[:(2*h+1)] = currentGuessCosSin[:(2*h+1)]
            gesIdx = (2*h+1)
        
        hbCoeffs[(2*h+1):] = currentGuessCosSin[gesIdx:-1]
        
        omega = currentGuessCosSin[-1]
        
        return hbCoeffs, omega
    
    def calculateHB(self, 
                    currenteGuessCosSin: np.ndarray, 
                    sample: float):
        r"""
        Calculates the HB solution of the electrical system of the beta cell

        Parameters
        ----------
        currenteGuessCosSin : np.ndarray
            Current guess for the FgPC coefficients.
        sample : float
            current sample

        Returns
        ----------
        residuum : np.ndarray
            Residuum of the electrical system of the beta cell
        """
        
        t0_total = time.time()
        fgpcCoeff, omega = self.splitCoeffVec(currenteGuessCosSin)

        pos, vel = self.getPosVel(fgpcCoeff, omega)

        timeRes = self.calculateElectricResiduum(pos, vel, [sample])
        
        residuum = self.calcFreqResiduum(timeRes)

        t1_total = time.time()
        self.itrCt += 1
        print("Iteration: " + str(self.itrCt) + 
              " Function Eval Time: " + '{:.8f}'.format(t1_total-t0_total) +
              " Residuum Norm: " + '{:.6e}'.format(np.sqrt(np.sum(residuum**2))), end='\r')
        return residuum


    def calculateFgPC(self, currentGuessCosSin, quadPt, polyVals):
        r"""
        Calculates the time residuum of the electrical system 
        of the beta cell.

        Parameters
        ----------
        currentGuessCosSin : np.ndarray
            Current guess for the FgPC coefficients.
        quadPt : np.ndarray
            quadrature point
        polyVals : np.ndarray
            Polynomial values at quadrature point.
        
        Returns
        ----------
        timeResiduum : np.ndarray
            Time residuum of the electrical system of the beta cell.
        """
        
        FgPCCoeffs, omegaCoeffs = self.constructCosSinFgPCCoeffs(currentGuessCosSin)

        pos, vel = self.getPosVel(FgPCCoeffs, omegaCoeffs, polyVals)
        
        timeRes = self.calculateElectricResiduum(pos, vel, quadPt)
        
        return timeRes

    def calculateElectricResiduum(self, 
                                  pos: np.ndarray, 
                                  vel: np.ndarray, 
                                  quadPt: np.ndarray):
        r"""
        Method to calculate the residuum of the electric beta cell model.
        
        Parameters
        ----------
        pos : np.ndarray
            The position of V, n and Ca of the electrical system .
        vel : np.ndarray
            The velocity of V, n and Ca of the electrical system.
        quadPt : np.ndarray
            The evaluation points for the quadrature rule.

        Returns
        ----------
        residuum : np.ndarray
            The residuum of the electric beta cell model,
            with the residuum of V, n, and Ca stacked.
        """
        
        v = pos[0:self.nrEvalPts]
        n = pos[self.nrEvalPts:2*self.nrEvalPts]
        ca = pos[2*self.nrEvalPts:3*self.nrEvalPts]

        dvdt = vel[0:self.nrEvalPts]/1000 # V/ms
        dndt = vel[self.nrEvalPts:2*self.nrEvalPts]/1000 # 1/ms
        dcadt = vel[2*self.nrEvalPts:3*self.nrEvalPts]/1000 # µM/ms

        ca_er = self.betaCell.Ca_er # µM
        atp = quadPt[0]
        adp = self.betaCell.concentrationADP(atp) # µM

        i_ca, i_k, i_kca, i_katp = \
            self.betaCell.calculateIonCurrent(v, n, ca, adp, atp)

        n_inf = self.betaCell.activationFun(v, self.betaCell.n_in, \
                        self.betaCell.s_n)
        
        j_mem = self.betaCell.calcJmem(i_ca, ca)
        j_er = self.betaCell.calcJer(ca, ca_er)

        resTimeV = dvdt + 1/self.betaCell.C_mem * (i_ca + i_k + i_kca + i_katp)
        resTimen = dndt - (n_inf-n)/self.betaCell.tau_n
        resTimeCa = dcadt - self.betaCell.f_Ca*(j_mem-j_er)
        
        if self.sinMt == 1:
            residuum = np.concatenate((resTimeV, resTimen, resTimeCa))
        elif self.sinMt == 2:
            residuum = np.concatenate((resTimeV, resTimen, resTimeCa, dvdt[0]))

        return residuum
    
    def getPosVel(self,
                  FgPCCoeffs: np.ndarray, 
                  omegaCoeffs: np.ndarray, 
                  polyVals: np.ndarray = None):
        r"""
        Method to calculate the position, velocity and acceleration of the
        FgPC model of the electrical system
        
        Parameters
        ----------
        FgPCCoeffs : np.ndarray
            current FgPC coefficients
        omegaCoeffs : np.ndarray
            The omega gPC coefficients.
        polyVals: np.ndarray
            Polynomial values at quadrature point.

        Returns
        ----------
        pos and vel are stacked as following [V, n, Ca]
        pos : np.ndarray
            The position of the electrical system,
        vel : np.ndarray
            The velocity of the electrical system
        """
        
        ident = np.eye(self.totalH)
        if polyVals is None:
            hbCoeffs = FgPCCoeffs
            omega = omegaCoeffs
        else:
            hbCoeffs = np.dot(np.kron(ident,polyVals), FgPCCoeffs)
            omega = np.dot(polyVals, omegaCoeffs)

        hbCoeffs_c = self.convertAllHBSinCos2Complx(hbCoeffs)

        pos = self.calcPosition(hbCoeffs_c, self.E_nh_c_total)
        vel = self.calcVelocity(hbCoeffs_c, self.E_nh_c_total,
                                omega, self.derMat_c_total)

        return pos, vel
