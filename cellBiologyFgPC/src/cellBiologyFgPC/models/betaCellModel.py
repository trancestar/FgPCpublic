r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    This class contains the related model functions for 
                the beta cell model.
"""

import numpy as np

class betaCellfunctions:

    def __init__(self) -> None:
        # set units according to paper (?Bertrand, see Clasen)
        m = 1
        mu = 1
        n =1 
        p = 1
        f =1

        # set electric parameters
        # membrane and Ca components (V, n, Ca, Ca_er)
        self.C_mem = 5300*f # F, membrane capacity
        self.tau_n = 20*m # ms, time constant rectifying current
        self.s_n = 5*m # 1, shape parameter k
        self.n_in = -16*m # 1, shape parameter rectifying current /v_n
        self.V_K = -75*m # mV, nernst potential K+
        self.V_Ca = 25*m # mV, nernst potential Ca2+

        # I_Ca
        self.g_CA = 1000*p # S(?) maximal conductance ca membran channel
        self.s_m = 12*m # 1, shape parameter Ca2+ activation
        self.u_m = -20*m # 1, shape parameter Ca2+ activation

        # I_K
        self.g_K = 2700*p # S(?) maximal conductance of voltage dependent K+ channel

        # I_KCA
        self.g_KCA = 100*p # S(?) maximal conductance of Ca2+ dependent K+ channel
        self.k_d = 0.5*mu # MGADP^2 factor KATP activation function

        # I_KATP
        self.g_KATP = 26000*p # S(?) maximal conductance of ATP sensitive K+ channel

        # Calcium Handling
        self.alpha = 5.18e-18*mu/(f*m) # current to ion flux conversion factor
        self.V_cyt = 1.15e-12 # L, cytosol volume
        self.k_PMCA = 0.2/m # s^-1, pmca pump rate ca2+ cell membrane
        self.f_Ca = 0.01 # 1, fraction of ca2+ not bound to buffers

        self.p_leak = 2.0e-4*m # ?, leak rate of ca2+ from er to cytosol over er membran
        self.sigma_er = 31 # 1, ratio of ER to cytosol volume cyt volume/ER volume
        self.k_SERCA = 0.4/m # s^-1, SERCA pump rate ca2+ from cytosol to er

        # KATP channel 
        self.k_dd = 17*mu # MgADP factor KATP activation function
        self.k_td = 26*mu # ADP3- factor KATP activation function
        self.k_tt = 1*mu # ATP4- factor KATP activation function

        self.A_tot = 3000 # µM, total nucleotide concentration

        # set constant when uncouple metabolic and electric system
        self.c = 0.07 # µM, Ca2+ concentration
        self.atp = 2000*mu

        # electrical system constants
        self.Ca_er = 341*mu # µM, Ca2+ concentration in the ER
    
            
    def calcJmem(self, 
                 i_ca: float, 
                 ca: float):
        r"""
        Method calculates the current from Ca2+ over the membrane
        
        Parameters:
        -----------
        i_ca : float
            Ca2+ current
        ca : float
            concentration of Ca2+

        Returns:
        -----------
        j_mem : float
            current from Ca2+ over the membrane
        """
        return -(self.alpha/self.V_cyt*i_ca + self.k_PMCA*ca)
    
    def calcJer(self, 
                ca: float, 
                ca_er: float):
        r"""
        Method calculates the current from Ca2+ into the ER
        
        Parameters:
        -----------
        ca : float
            concentration of Ca2+
        ca_er : float
            concentration of Ca2+ in the ER

        Returns:
        -----------
        j_er : float
            current from Ca2+ into the ER
        """
        return self.k_SERCA* ca - self.p_leak* (ca_er-ca)
        
    def calculateIonCurrent(self, 
                            v: float, 
                            n: float, 
                            ca: float, 
                            adp:float, 
                            atp: float):
        r"""
        Method calculates the ion currents

        Parameters:
        -----------
        v : float
            membrane potential
        n : float
            activation variable of the K+ current
        ca : float
            concentration of Ca2+
        adp : float
            concentration of adenosindiphosphat
        atp : float
            concentration of adenosintriphosphat

        Returns:
        -----------
        i_ca : float
            Ca2+ current
        i_k : float
            K+ current
        i_kca : float
            K+ current dependent on Ca2+
        i_katp : float
            K+ current dependent on ATP
        """

        i_ca = self.calcICa(v)
        i_k = self.calcIk(v, n)
        i_kca = self.calcIKCa(ca,v)
        i_katp = self.calcIKATP(v, adp, atp)

        return i_ca, i_k, i_kca, i_katp
    
    def calcIKCa(self, 
                 ca: float, 
                 v: float):
        r"""
        Method calculates and returns the K+ current

        Parameters:
        -----------
        ca : float
            concentration of Ca2+
        v : float
            membrane potential
        
        Returns:
        -----------
        i_kca : float
            K+ current
        """
        openingCaChannel = ca**2/(ca**2+self.k_d**2)
        return self.g_KCA* openingCaChannel* (v-self.V_K)
    
    def calcIKATP(self, 
                  v: float, 
                  adp: float, 
                  atp: float):
        r"""
        Method calculates and returns the KATP current

        Parameters:
        -----------
        v : float
            membrane potential
        adp : float
            concentration of adenosindiphosphat
        atp : float
            concentration of adenosintriphosphat
        
        Returns:
        -----------
        i_katp : float
            KATP current
        """
        o_inf = self.activationFunKATP(adp, atp)
        return self.g_KATP*(v-self.V_K)* o_inf
    
    def activationFunKATP(self, 
                          adp: float, 
                          atp: float):
        r"""
        Method calculates and returns the activation function 
        of the KATP current

        Parameters:	
        -----------
        adp : float
            concentration of adenosindiphosphat
        atp : float
            concentration of adenosintriphosphat
    
        Returns:
        -----------
        o_inf : float
            activation function of the KATP current
        """
        mgADP = 0.165*adp
        adp3 = 0.135*adp
        atp4 = 0.05*atp

        o1 = 1+ 2*mgADP/self.k_dd
        o2 = mgADP/self.k_dd
        o3 = 1+ mgADP/self.k_dd
        o4 = 1+ atp4/self.k_tt + adp3/self.k_td

        return (0.08*o1+ 0.89* o2**2)/(o3**2 * o4)
        
    def calcIk(self, 
               v: float, 
               n: float):
        r"""
        Method calculates and returns the K+ current

        Parameters:
        -----------
        v : float
            membrane potential
        n : float
            activation variable of the K+ current
        
        Returns:
        -----------
        i_k : float
            K+ current
        """
        return self.g_K*n*(v-self.V_K)
    
    def calcICa(self, 
                v: float):
        r"""
        Method calculates and returns the Ca2+ current

        Parameters:
        -----------
        v : float
            membrane potential
        
        Returns:
        -----------
        i_ca : float
            Ca2+ current
        """
        m_inf = self.activationFun(v, self.u_m, self.s_m)
        return self.g_CA* m_inf* (v-self.V_Ca)

    def activationFun(self, 
                      v: float, 
                      nu: float, 
                      s: float):
        r"""
        Method calculates and returns the activation function 
        of the Ca2+ current

        Parameters:	
        -----------
        v : float
            membrane potential
        nu : float
            shape parameter
        s : float
            shape parameter
    
        Returns:
        -----------
        m_inf : float
            activation function of the Ca2+ current
        """
        return 1/(1+np.exp((nu-v)/s))

    def concentrationADP(self, 
                         atp: float):
        r"""
        Returns the ATP concentration in µM.

        Parameters:
        -----------
        atp : float
            concentration of adenosintriphosphat
        
        Returns:
        -----------
        adp : float
            concentration of adenosindiphosphat
        """
        return -0.5* atp+ np.sqrt(atp**2/4+self.A_tot*atp-atp**2)


    