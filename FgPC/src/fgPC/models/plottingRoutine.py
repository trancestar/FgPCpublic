r"""
Author:         Lars de Jong
Date:           2024-05-15
Description:    Handles the plotting of the results of the FgPC and MC model.

"""
import numpy as np
import copy
import pickle

from fgPC.utils.plotlib import Plotter, LinePlot, HistPlot, ImagePlot

class plottingRoutine:
    r"""
    Class contains all important methods to calculate the plots
    """

    def __init__(self):
        pass

    def plottingResults(self, 
                        curFgPCResult: list, 
                        curmcResult: list,
                        curFgPCStochastics: list, 
                        curdiffStochastics: list, 
                        timeVec: np.ndarray, 
                        idxTime: int, 
                        histTimeList: list, 
                        samplingTimeIdxList: list, 
                        coveraStr: str, 
                        plotter: Plotter,
                        varStr: str, 
                        figSize: tuple = (16.5, 9),
                        xlabel: str = "time", 
                        xticksList: list = None,
                        velPlot: bool = False, 
                        diffPlot: bool = False,
                        sampleStr: list = None,
                        msize: int = 7):
        r"""
        Method takes care of all plotting  
        of each given variable regarding the position over time, 
        velocity over time, model differenc over time, marginal 
        distribution at given time points as well as the input 
        distribution

        Parameters
        ----------
        curFgPCResult : list
            list with sampling of the FgPC model at the given time points
        curmcResult : list
            list with sampling of the MC model at the given time points
        curFgPCStochastics : list
            list with the mean, lower and upper bound of the FgPC model 
            over one period
        curdiffStochastics : list
            list with the mean, lower and upper bound of the difference
            between the FgPC and MC model over one period
        timeVec : np.ndarray
            time vector with time points over one period
        idxTime : int
            index of the time point to be plotted
        histTimeList : list
            list with time points for the marginal distribution
        samplingTimeIdxList : list
            list with the indices for mean, lower and upper bound 
            for the time points of the marginal distribution
        coveraStr : str
            string with the coverage of the confidence interval
        plotter : Plotter
            object of the Plotter class to handle the plotting
        varStr : str
            string with the name of the variable to be plotted
        figSize : tuple, optional
            size of the figure, by default (16.5, 9)
        xlabel : str, optional
            label of the x-axis, by default "Time"
        xticksList : list, optional
            list with the x-ticks and labels, by default None
        velPlot : bool, optional
            flag to plot the velocity over time, by default False
        diffPlot : bool, optional
            flag to plot the difference between the FgPC and MC 
            model, by default False
        sampleStr : list, optional
            list with sampleStr and the uncertain variable string
            for the input distribution, by default None
        msize : int, optional
            size of the markers, by default 7
        """

        # plot time plots
        xPlots = self.getTimePlots(timeVec, 
                                   [curFgPCStochastics[0], 
                                    curFgPCStochastics[2], 
                                    curFgPCStochastics[3]], 
                                   idxTime, 
                                   coveraStr,
                                   xlabel = xlabel,
                                   xticksList = xticksList,
                                   ylabel = varStr,
                                   msize = msize)
        
        plotter.plot(xPlots, 
                     filename = varStr[1] + "_posPlots",
                     fig_size = figSize)
        

        if velPlot:
            vPlots = self.getTimePlots(timeVec, 
                                       [curFgPCStochastics[1], 
                                        curFgPCStochastics[4], 
                                        curFgPCStochastics[5]], 
                                       idxTime, 
                                       coveraStr,
                                       xlabel = xlabel,
                                       xticksList = xticksList,
                                       ylabel = 'd/dt ' + varStr,
                                       msize = msize)
            plotter.plot(vPlots, 
                         filename = varStr[1] + "_velPlots",
                         fig_size = figSize)
        

        if diffPlot:
            diffPlots = self.getDiffPlots(timeVec, 
                                          [curdiffStochastics[0],
                                           curdiffStochastics[2],
                                           curdiffStochastics[3]], 
                                          ["mean", coveraStr + r" \%",# Samples",
                                           coveraStr + r" \%" ],#Samples"],
                                           xlabel = xlabel,
                                           xticksList = xticksList)
            plotter.plot(diffPlots, 
                         filename = varStr[1] + "_posDiffPlots",
                         fig_size = figSize)
            
        for timeTarget,idxTarget,curFgPCSampling,curmcSampling in zip(histTimeList,samplingTimeIdxList,curFgPCResult,curmcResult):

            timeStr ="distribution of " + varStr + " at time " + '{:.2f}'.format(timeVec[idxTarget])
            dist_plots = self.histoPlots(curFgPCSampling, 
                                         curmcSampling, 
                                         [curFgPCStochastics[0][idxTarget], 
                                          curFgPCStochastics[2][idxTarget],
                                          curFgPCStochastics[3][idxTarget]],
                                         timeStr)
            plotter.plot(dist_plots, 
                         filename = varStr[1] + "_distPlots_at_" + str(timeTarget),
                         fig_size = figSize)
            
        if sampleStr is not None:
            with open(sampleStr[0], "rb") as f:
                samples = pickle.load(f)

            _, bin_edges = np.histogram(samples, bins='fd')
            samplesPlot = HistPlot(samples, sampleStr[1], "nr. samples", 
                                    bins = bin_edges, color='red')
            
            plotter.plot(samplesPlot,
                         filename = "Input_Distribution",
                         fig_size = figSize)


    def getPhasePlots(self,
                      FgPCStochVar1: list, 
                      FgPCStochVar2: list,
                      varStr1: str, 
                      varStr2: str, 
                      idxTime: int,
                      coveraStr: str, 
                      plotter: Plotter, 
                      timeVec: np.ndarray,
                      figSize: tuple = (13, 9), 
                      xlabel: str = "time",
                      xticksList: list = None,
                      msize: int = 7, 
                      xlim: tuple = None, 
                      ylim: tuple = None):
        r"""
        Method to plot the phase plot of two variables

        Parameters
        ----------
        FgPCStochVar1 : list
            list with the mean, lower and upper bound of the first variable
        FgPCStochVar2 : list
            list with the mean, lower and upper bound of the second variable
        varStr1 : str
            string with the name of the first variable
        varStr2 : str
            string with the name of the second variable
        idxTime : int
            index of the time point to be plotted
        coveraStr : str
            string with the coverage of the confidence interval
        plotter : Plotter
            object of the Plotter class to handle the plotting
        timeVec : np.ndarray
            time vector with time points of one period
        figSize : tuple, optional
            size of the figure, by default (13, 9)
        xlabel : str, optional
            label of the x-axis, by default "Time"
        xticksList : list, optional
            list with the x-ticks and labels, by default None
        msize : int, optional
            size of the markers, by default 7
        xlim : tuple, optional
            limits of the x-axis, by default None
        ylim : tuple, optional
            limits of the y-axis, by default None
        """
        
        FgPCPhaseMean = LinePlot(FgPCStochVar1[0],FgPCStochVar2[0],varStr1,varStr2,
                                color = "black", label = "FgPC mean", linestyle = "-")
        FgPCPhaseLow = LinePlot(FgPCStochVar1[1],FgPCStochVar2[1],varStr1,varStr2,
                                color = "red", label = coveraStr + r" \% samples", linestyle = "--") 
        FgPCPhaseHigh = LinePlot(FgPCStochVar1[2],FgPCStochVar2[2],varStr1,varStr2,
                                color = "blue", linestyle = "--")
        
        PhasePlots = [FgPCPhaseMean, FgPCPhaseLow, FgPCPhaseHigh]
        
        var1Plots = self.getVertTimePlots(timeVec,
                                          FgPCStochVar1,
                                          idxTime,
                                          coveraStr,
                                          xlabel = xlabel,
                                          yticksList = xticksList,
                                          ylabel = varStr1,
                                          msize = msize)
        
        var2Plots = self.getTimePlots(timeVec,
                                      FgPCStochVar2,
                                      idxTime,
                                      coveraStr,
                                      xlabel = xlabel,
                                      xticksList = xticksList,
                                      ylabel = varStr2)
        
        helpPlot = LinePlot([],[],"","")

        for i in range(3,len(var1Plots)-1):
            del var1Plots[i].kwargs['label']
            del PhasePlots[i].kwargs['label']

        plotter.plot(helpPlot,var2Plots,var1Plots,PhasePlots,
                     filename = varStr1[1] + "_" + varStr2[1] + "_phasePlot",
                     fig_size = figSize)
        
        if xlim is not None:
            PhasePlotsZoom = copy.deepcopy(PhasePlots)
            for plot in PhasePlotsZoom:
                plot.xlim = xlim
                plot.ylim = ylim
            plotter.plot(PhasePlotsZoom,
                         filename = varStr1[1] + "_" + varStr2[1] + "_phasePlotZoom",
                         fig_size = figSize)


    def getTimePlots(self, 
                     time: np.ndarray, 
                     curFgPCStochastics: list, 
                     idxTime: int,
                     coveraStr: str, 
                     xlabel: str = "time",
                     xticksList: list = None,
                     ylabel: str = 'y',
                     msize: int = 10):
        r"""
        Method to get the variable over time plots

        Parameters
        ----------
        time : np.ndarray
            time vector with time points over one period
        curFgPCStochastics : list
            list with the mean, lower and upper bound of the variable
        idxTime : int
            index of the time point to be plotted
        coveraStr : str
            string with the coverage of the confidence interval
        xlabel : str, optional
            label of the x-axis, by default "Time"
        xticksList : list, optional
            list with the x-ticks and labels, by default None
        ylabel : str, optional
            label of the y-axis, by default 'y'
        msize : int, optional
            size of the markers, by default 10
        
        Returns
        ----------
        xPlots : list
            list with the LinePlot objects
        """

        FgPCxMean = LinePlot(time, curFgPCStochastics[0], xlabel, ylabel,
                        color = "black", label = "FgPC mean", linestyle = "-")
        FgPCxLow = LinePlot(time, curFgPCStochastics[1], xlabel, ylabel,
                            color = "red", label = coveraStr + r" \% samples", linestyle = "--")
        FgPCxHigh = LinePlot(time, curFgPCStochastics[2], xlabel, ylabel,
                            color = "blue", linestyle = "--")
        
        xPlots = [FgPCxMean, FgPCxLow, FgPCxHigh]
        
        if xticksList is not None:
            for plot in xPlots:
                plot.xticksList = xticksList[0]
                plot.xticksLabelList = xticksList[1]

        return xPlots
    
    def getVertTimePlots(self, 
                         time: np.ndarray, 
                         curFgPCStochastics: list, 
                         idxTime: int, 
                         coveraStr: str, 
                         xlabel: str = "time",
                         ylabel: str = 'y', 
                         yticksList: list = None,
                         msize: int = 10):
        r"""
        Method to make a plot wiht the time over variable

        Parameters
        ----------
        time : np.ndarray
            time vector with time points over one period
        curFgPCStochastics : list
            list with the mean, lower and upper bound of the variable
        idxTime : int
            index of the time point to be plotted
        coveraStr : str
            string with the coverage of the confidence interval
        xlabel : str, optional
            label of the x-axis, by default "Time"
        ylabel : str, optional
            label of the y-axis, by default 'y'
        yticksList : list, optional
            list with the y-ticks and labels, by default None
        msize : int, optional
            size of the markers, by default 10
        
        Returns
        ----------
        xPlots : list
            list with the LinePlot objects
        """

        FgPCxMean = LinePlot(curFgPCStochastics[0], time, ylabel, xlabel,
                        color = "black", label = "FgPC mean", linestyle = "-")
        FgPCxLow = LinePlot(curFgPCStochastics[1], time, ylabel, xlabel,
                            color = "red", label = coveraStr + r" \% samples", linestyle = "--")
        FgPCxHigh = LinePlot(curFgPCStochastics[2], time, ylabel, xlabel,
                            color = "blue", linestyle = "--")
        
        xPlots = [FgPCxMean, FgPCxLow, FgPCxHigh]

        if yticksList is not None:
            for plot in xPlots:
                plot.yticksList = yticksList[0]
                plot.yticksLabelList = yticksList[1]
        
        return xPlots

    def getDiffPlots(self, 
                     time: np.ndarray, 
                     diffStoch: list, 
                     labels: str, 
                     xlabel: str = "time", 
                     xticksList: list = None):
        r"""
        Method to plot the difference between the FgPC and MC model 
        over one period.

        Parameters
        ----------
        time : np.ndarray
            time vector with time points over one period
        diffStoch : list
            list with the mean, lower and upper bound of the difference
            between the FgPC and MC model over one period
        labels : str
            string with the labels for the plot
        xlabel : str, optional
            label of the x-axis, by default "Time"
        xticksList : list, optional
            list with the x-ticks and labels, by default None
        
        Returns
        ----------
        diffPlotList : list
            list with the LinePlot objects
        """

        colorList = ["black", "red", "blue"]
        lineStyleList = ['-','--','--']
        diffPlotList = []
        for idx in range(len(diffStoch)):
            diffPlot = LinePlot(time, diffStoch[idx], xlabel, "difference",
                                color = colorList[idx],
                                linestyle = lineStyleList[idx],
                                label = labels[idx])
            if xticksList is not None:
                diffPlot.xticksList = xticksList[0]
                diffPlot.xticksLabelList = xticksList[1]
            diffPlot.y_scale = 'log'
            diffPlotList.append(diffPlot)

        return diffPlotList
    
    def histoPlots(self, 
                   FgPCSamples: np.ndarray, 
                   mcSamples: np.ndarray, 
                   stochastics: list, 
                   xLabel: str):
        r"""
        Method to plot the marginal distribution of the FgPC and MC model

        Parameters
        ----------
        FgPCSamples : np.ndarray
            samples of the FgPC model
        mcSamples : np.ndarray
            samples of the MC model
        stochastics : list
            list with the mean, lower and upper bound of the variable
        xLabel : str
            label of the x-axis
        
        Returns
        ----------
        dist_plots : list
            list with the HistPlot objects
        """

        mean = stochastics[0]
        lowBd = stochastics[1]
        upBd = stochastics[2]


        _, bin_edges = np.histogram(np.concatenate([FgPCSamples,mcSamples]), bins='fd')
        
        FgPC_histPlot = HistPlot(FgPCSamples, xLabel, "nr. samples", 
                                vline=[mean, upBd, lowBd],
                                bins = bin_edges, color='blue', 
                                alpha=1.0, label='FgPC')
        mc_histPlot = HistPlot(mcSamples, xLabel, "nr. samples", 
                            bins = bin_edges, color='red',
                            alpha=0.5, label='MC')
        
        return [FgPC_histPlot, mc_histPlot]
    
    def plotCoeffMap(self,
                     coeffVec: np.ndarray,
                     h: int,
                     ngPC: int,
                     varNr: int,
                     varList: list,
                     forced: bool,
                     plotter: Plotter,
                     figSize: tuple = (16.5, 9)):
        r"""
        Method to plot the coefficient maps of the FgPC solution

        Parameters
        ----------
        coeffVec : np.ndarray
            vector with the coefficients
        h : int
            number of harmonics
        ngPC : int
            degree of polynomials
        varNr : int
            number of variables
        varList : list
            list with the names of the variables
        forced : bool
            boolean to indicate if the system is forced
        plotter : Plotter
            object of the Plotter class to handle the plotting
        figSize : tuple, optional
            size of the figure, by default (16.5, 9)
        """

        # make coeff tensor
        coeffTensor = self.getCoeffTensor(coeffVec, h, ngPC, varNr, forced)

        # plot coeff tensor
        
        xticksList = [0]
        xticksLabelList = [str(0)]
        for i in range(1,h+1,2):
            xticksLabelList.append(str(i))
            xticksList.append(i)
        
        for idxVar in range(varNr):

            coeffs = np.abs(coeffTensor[idxVar,:,:])
            if not forced and idxVar == 0:
                omegaVar = np.abs(coeffVec[-ngPC:])
                minVal = min(np.min(coeffs), np.min(omegaVar))
                maxVal = max(np.max(coeffs), np.max(omegaVar))
                coeffs = np.vstack((coeffs, np.zeros(ngPC),omegaVar))

                xticksList.append(coeffs.shape[0]-1)
                xticksLabelList.append(r"$\omega$")

                xlabel = r"harmonics $H$ of $" + varList[idxVar] + r"$ and base frequency $\omega$"
            else:
                minVal = np.min(coeffs)
                maxVal = np.max(coeffs)
                xlabel = r"harmonics $H$ of $" + varList[idxVar] + r"$"

            coeffPlot = ImagePlot(coeffs.T, xlabel, r"polynomial degree $N$",
                                  cbarLabel = r"$\sqrt{a_{km}^2 + b_{km}^2}$",
                                  barScale = ["log", minVal, maxVal],
                                  gridBoundary = True)
            
            coeffPlot.xticksList = xticksList
            coeffPlot.xticksLabelList = xticksLabelList

            plotter.plot(coeffPlot,
                         filename= varList[idxVar] + "_coeffMap",
                         fig_size=figSize)
            
            if not forced and idxVar == 0:
                xticksList.pop()
                xticksLabelList.pop()


    def getCoeffTensor(self,
                       coeffs: np.ndarray,
                       h: int,
                       ngPC: int,
                       varNr: int,
                       forced: bool):

        coeffTensor = np.zeros((varNr, h+1, ngPC))
        # coeffTensor = np.zeros((varNr, 2*h+1, ngPC))

        baseIdx = 0
        for idxVar in range(varNr):

            coeffTensor[idxVar, 0, :] = coeffs[baseIdx:baseIdx+ngPC]
            baseIdx += ngPC

            cosCoeffVar = coeffs[baseIdx:baseIdx+ngPC*h]
            baseIdx += ngPC*h

            if not forced and idxVar == 0:
                sinCoeffVar = coeffs[baseIdx:baseIdx+ngPC*(h-1)]
                baseIdx += ngPC*(h-1)
            else:
                sinCoeffVar = coeffs[baseIdx:baseIdx+ngPC*h]
                baseIdx += ngPC*h
            
            baseH = 0
            baseSinH = 0
            # curH = 1
            for idxH in range(h):

                cosCoeffH = cosCoeffVar[baseH:baseH+ngPC]
                
                if not forced and idxVar == 0 and idxH == 0:
                    sinCoeffH = np.zeros(ngPC)
                else:
                    sinCoeffH = sinCoeffVar[baseSinH:baseSinH+ngPC]
                    baseSinH += ngPC

                coeffTensor[idxVar, idxH+1, :] = np.sqrt(cosCoeffH**2 + sinCoeffH**2)
                baseH += ngPC

        return coeffTensor
    
    def plotTotalConvergenceMap(self,
                                errorMat: np.ndarray,
                                tag: str,
                                plotter: Plotter,
                                figSize: tuple = (16.5, 9)):
        r"""
        Method to plot the convergence map of the FgPC model

        Parameters
        ----------
        errorMat : np.ndarray
            matrix with the error values, 
            axis 0 is the harmonics, axis 1 is the ngPC
        varList : list
            list with the names of the variables
        plotter : Plotter
            object of the Plotter class to handle the plotting
        figSize : tuple, optional
            size of the figure, by default (16.5, 9)
        """

        xticksList = []
        xticksLabelList = []
        for i in range(1,errorMat.shape[0]+1,2):
            xticksLabelList.append(str(i))
            xticksList.append(i-1)

        yticksList = []
        yticksLabelList = []
        for i in range(0,errorMat.shape[1],2):
            yticksLabelList.append(str(i))
            yticksList.append(i)

        errorMat = np.nan_to_num(errorMat, nan = 0)
        if np.min(errorMat) == 0:
            mask_errorMat = errorMat[errorMat != 0]
            minVal = mask_errorMat.min()
        else:
            minVal = np.min(errorMat)
        maxVal = np.max(errorMat)

        xlabel = r"harmonics $H$"

        erorPlot = ImagePlot(errorMat.T, xlabel, r"polynomial degree $N$",
                                cbarLabel = r"$\varepsilon_{km}$",
                                barScale = ["log",minVal, maxVal],
                                gridBoundary = True)
        
        erorPlot.xticksList = xticksList
        erorPlot.xticksLabelList = xticksLabelList
        erorPlot.yticksList = yticksList
        erorPlot.yticksLabelList = yticksLabelList

        plotter.plot(erorPlot,
                        filename= tag + "_errorMap",
                        fig_size=figSize)

    def plotConvergenceMap(self,
                           errorTensor: np.ndarray,
                           varList: list,
                           plotter: Plotter,
                           figSize: tuple = (16.5, 9)):
        r"""
        Method to plot the convergence map of the FgPC model

        Parameters
        ----------
        errorTensor : np.ndarray
            tensor with the error values, 
            axis 0 is the variable, axis 1 is the harmonic,
            axis 2 is the ngPC
        varList : list
            list with the names of the variables
        plotter : Plotter
            object of the Plotter class to handle the plotting
        figSize : tuple, optional
            size of the figure, by default (16.5, 9)
        """

        xticksList = []
        xticksLabelList = []
        for i in range(1,errorTensor.shape[1]+1,2):
            xticksLabelList.append(str(i))
            xticksList.append(i-1)

        yticksList = []
        yticksLabelList = []
        for i in range(0,errorTensor.shape[2],2):
            yticksLabelList.append(str(i))
            yticksList.append(i-1)

        for i in range(errorTensor.shape[0]):

            errorMat = errorTensor[i,:,:]
            errorMat = np.nan_to_num(errorMat, nan = 0)
            if np.min(errorMat) == 0:
                mask_errorMat = errorMat[errorMat != 0]
                minVal = mask_errorMat.min()
            else:
                minVal = np.min(errorMat)
            maxVal = np.max(errorMat)

            xlabel = r"Harmonics $H$ of " + varList[i]

            erorPlot = ImagePlot(errorMat.T, xlabel, r"Polynomial degree $N$",
                                 cbarLabel = r"$\varepsilon_{km}$",
                                 barScale = ["log",minVal, maxVal],
                                 gridBoundary = True)
            
            erorPlot.xticksList = xticksList
            erorPlot.xticksLabelList = xticksLabelList
            erorPlot.yticksList = yticksList
            erorPlot.yticksLabelList = yticksLabelList

            plotter.plot(erorPlot,
                         filename= varList[i] + "_errorMap",
                         fig_size=figSize)

                
        
