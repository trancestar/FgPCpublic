r"""
Author:         Lars de Jong
Date:           2024-05-15
Description:    Handles the plotting of the results of the FgPC and MC model.

"""
import numpy as np
import copy
import pickle
import sys

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
                        histTimeList: list, 
                        samplingTimeIdxList: list, 
                        plotter: Plotter,
                        varStr: str, 
                        legendStr: list,
                        figSize: tuple = (16.5, 9),
                        xlabel: str = r"$t$", 
                        xticksList: list = None,
                        velPlot: bool = False, 
                        diffPlot: bool = False,
                        sampleStr: list = None):
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
        histTimeList : list
            list with time points for the marginal distribution
        samplingTimeIdxList : list
            list with the indices for mean, lower and upper bound 
            for the time points of the marginal distribution
        plotter : Plotter
            object of the Plotter class to handle the plotting
        varStr : str
            string with the name of the variable to be plotted
        legendStr : list
            list with the names of the variables
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
        """

        # plot time plots
        xPlots = self.getTimePlots(timeVec, 
                                   [curFgPCStochastics[0], 
                                    curFgPCStochastics[2], 
                                    curFgPCStochastics[3]], 
                                   [r"$\mathbb{E}[" + legendStr[0]+r"(t,"+ legendStr[1] + ")]$",
                                    r"$Q_{" + legendStr[0]+r"(t,"+ legendStr[1] + ")}(0.025)$",
                                    r"$Q_{" + legendStr[0]+r"(t,"+ legendStr[1] + ")}(0.975)$"],
                                   xlabel = xlabel,
                                   xticksList = xticksList,
                                   ylabel = varStr)
        
        plotter.plot(xPlots, 
                     filename = varStr[1] + "_posPlots",
                     fig_size = figSize)
        

        if velPlot:
            vPlots = self.getTimePlots(timeVec, 
                                       [curFgPCStochastics[1], 
                                        curFgPCStochastics[4], 
                                        curFgPCStochastics[5]], 
                                       [r"$\mathbb{E}[" + legendStr[2]+r"(t,"+ legendStr[3] + ")]$",
                                        r"$Q_{" + legendStr[2]+r"(t,"+ legendStr[3] + ")}(0.025)$",
                                        r"$Q_{" + legendStr[2]+r"(t,"+ legendStr[3] + ")}(0.975)$"],
                                       xlabel = xlabel,
                                       xticksList = xticksList,
                                       ylabel = 'd/dt ' + varStr)
            plotter.plot(vPlots, 
                         filename = varStr[1] + "_velPlots",
                         fig_size = figSize)
        

        if diffPlot:
            diffPlots = self.getDiffPlots(timeVec, 
                                          [curdiffStochastics[0],
                                           curdiffStochastics[1]], 
                                           [r"min${}_i (\varepsilon^{(i)})$",
                                            r"max${}_i (\varepsilon^{(i)})$"],
                                            legendStr[0],
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
                                   bins = bin_edges)
            
            plotter.plot(samplesPlot,
                         filename = "Input_Distribution",
                         fig_size = figSize)

    def getPhasePlots(self,
                       posSamples: np.ndarray,
                       velSamples: np.ndarray,
                       posMean: np.ndarray,
                       velMean: np.ndarray,
                       varStr1: str, 
                       varStr2: str,
                       labelList: list,
                       plotter: Plotter,
                       figSize: tuple = (13, 9)):
        r"""
        Method to plot the phase plot of the samples

        Parameters
        ----------
        posSamples : np.ndarray
            position samples of the first variable
        velSamples : np.ndarray
            velocity samples of the second variable
        posMean : np.ndarray
            mean of the position samples
        velMean : np.ndarray
            mean of the velocity samples
        varStr1 : str
            string with the name of the first variable
        varStr2 : str
            string with the name of the second variable
        labelList : list
            list with the names of the mean variables
        plotter : Plotter
            object of the Plotter class to handle the plotting
        figSize : tuple, optional
            size of the figure, by default (13, 9)
        """
        meanPlot = LinePlot(np.append(posMean,posMean[0]),
                            np.append(velMean,velMean[0]),
                            varStr1,varStr2, 
                            color = "red", linestyle = "-", 
                            label = labelList,
                            linewidth = 0.8)
        legendPlot = LinePlot(np.append(posMean,posMean[0]),
                            np.append(velMean,velMean[0]),
                            varStr1,varStr2,
                              color = "black", linestyle = "-", 
                              linewidth = 0.5, label = "samples")
        samplePlots = [legendPlot]
        for i in range(0,posSamples.shape[0]):
            samplePlot = LinePlot(np.append(posSamples[i,:],posSamples[i,0]),
                                np.append(velSamples[i,:],velSamples[i,0]),
                                varStr1,varStr2, color = "black", linestyle = "-", 
                                linewidth = 0.1, alpha = 0.3)
            samplePlots.append(samplePlot)
        samplePlots.append(meanPlot)
        
        plotter.plot(samplePlots,
                     filename = varStr1[1] + "_" + varStr2[1] + "_phasePlot",
                     fig_size = figSize)

    def getTimePlots(self, 
                     time: np.ndarray, 
                     curFgPCStochastics: list, 
                     labelList: list = None, 
                     xlabel: str = "time",
                     xticksList: list = None,
                     ylabel: str = 'y'):
        r"""
        Method to get the variable over time plots

        Parameters
        ----------
        time : np.ndarray
            time vector with time points over one period
        curFgPCStochastics : list
            list with the mean, lower and upper bound of the variable
        coveraStr : str
            string with the coverage of the confidence interval
        xlabel : str, optional
            label of the x-axis, by default "Time"
        xticksList : list, optional
            list with the x-ticks and labels, by default None
        ylabel : str, optional
            label of the y-axis, by default 'y'
        
        Returns
        ----------
        xPlots : list
            list with the LinePlot objects
        """

        if labelList is None:
            FgPCxMean = LinePlot(time, curFgPCStochastics[0], xlabel, ylabel,
                            color = "black", linestyle = "-")
            FgPCxLow = LinePlot(time, curFgPCStochastics[1], xlabel, ylabel,
                                color = "red", linestyle = "--")
            FgPCxHigh = LinePlot(time, curFgPCStochastics[2], xlabel, ylabel,
                                color = "blue", linestyle = "--")
        else:
            FgPCxMean = LinePlot(time, curFgPCStochastics[0], xlabel, ylabel,
                        color = "black", label = labelList[0], linestyle = "-")
            FgPCxLow = LinePlot(time, curFgPCStochastics[1], xlabel, ylabel,
                                color = "red", label = labelList[1], linestyle = "--")
            FgPCxHigh = LinePlot(time, curFgPCStochastics[2], xlabel, ylabel,
                                color = "blue", label = labelList[2], linestyle = "--")
            
        xPlots = [FgPCxMean, FgPCxLow, FgPCxHigh]
        
        if xticksList is not None:
            for plot in xPlots:
                plot.xticksList = xticksList[0]
                plot.xticksLabelList = xticksList[1]

        return xPlots

    def getDiffPlots(self, 
                     time: np.ndarray, 
                     diffStoch: list, 
                     labels: str, 
                     varStr: str,
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

        colorList = ["red", "blue"]
        lineStyleList = ['-','-']
        diffPlotList = []
        for idx in range(len(diffStoch)):
            # replace Zero with machine precision
            vals = diffStoch[idx]
            vals[np.abs(vals) <= sys.float_info.epsilon] = sys.float_info.epsilon        

            diffPlot = LinePlot(time, vals, 
                                xlabel, r"$\varepsilon^{(i)}(t)=|\tilde{" + varStr+ r"}^{(i)}(t)-" + varStr+ r"^{(i)}(t)|$",
                                color = colorList[idx],
                                linestyle = lineStyleList[idx],
                                linewidth = 1.0,
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
                                vline=[[mean, upBd, lowBd],
                                        ['black','blue','red']],
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

                
        