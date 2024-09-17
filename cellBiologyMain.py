r"""
Author:         Lars de Jong
Date:           2024-05-15
Description:    This is the main filefor the FgPC model of the
                electrical system of the beta cell.

"""
import os
import pickle
import time
import copy

import numpy as np

from fgPC.utils.filemanager import save_script
from fgPC.utils.logger import init_logger
from fgPC.utils.plotlib import Plotter
from fgPC.models.seekFgPCsolution import FgPCsolution
from fgPC.models.sampleModel import samplingModel
from fgPC.models.plottingRoutine import plottingRoutine
from fgPC.models.convergenceModel import convergenceFgPC

from cellBiologyFgPC.models.betaCellFgPC import betaCellFgPC
from cellBiologyFgPC.models.betaCellInitialGuess import BetaCellInitModel
from cellBiologyFgPC.utils.argparser import get_config_Integration, get_config_FgPC

# ---------------------
# --- Setup section ---
# ---------------------


deflationBool = False

convergenceStudy = True
confCoverage = 0.95

systemStr = "CellBiology"
varList = ["V", "n", "Ca"]

tag = "Electric"
tagList = [tag]

# set path where to store results
myResultPath = os.path.realpath(__file__) # use this when storing in same folder

saveResultsPath = myResultPath + "3_results/"

result_folder = save_script(saveResultsPath,
                            os.path.realpath(__file__), 
                            systemStr, tag, 
                            max_daily_folders = 7, 
                            max_res_folders = 7)

# logger
logger = init_logger(result_folder, "logfile")

# Get user input from: cmd line > config file > defaults
configFgPC = get_config_FgPC("./cellBiologyFgPC/cellFgPC.yaml", result_folder, logger)
configInt = get_config_Integration("./cellBiologyFgPC/cellTimeInt.yaml", result_folder, logger)

distributionStr = configFgPC.distStr + "_" + '{:.0f}'.format(configFgPC.low) + "_" \
        + '{:.0f}'.format(configFgPC.high)
solStr1 = "0_solutionData/" + tag + "_H"
solStr2 = "_" + distributionStr + ".pkl"
solStr = solStr1 + "{:02}".format(configFgPC.H) + "_gPC" + "{:02}".format(configFgPC.ngPC) \
    + solStr2

# ---------------------
# --- FgPC section ---
# ---------------------

# The FgPC objects handles the calculation of the FgPC coefficients
mySolutionObj = FgPCsolution(configFgPC, 
                                configInt,
                                myResultPath,
                                systemStr,
                                distributionStr,
                                tag,
                                varNr = len(varList), 
                                forced = False,
                                exptSol = 1,
                                logger = logger,
                                deflationBool = deflationBool)

startFgPCCoef = time.time()
if os.path.exists(solStr):
    with open(solStr, "rb") as f:
        FgPCSolList = pickle.load(f)
    
    myBetaCellFgPC = betaCellFgPC(configFgPC,
                                  tag,
                                  logger = logger)
    
else:

    myBetaCellFgPC = betaCellFgPC(configFgPC,
                                  tag,
                                  logger = logger)

    # calculates the FgPC coefficients, if deflation is set to True, 
    # all possible solutions are calcualted.
    FgPCSolList = mySolutionObj.calFgPCSolution(myBetaCellFgPC,
                                                BetaCellInitModel(1),
                                                configFgPC.H,
                                                configFgPC.ngPC)
    with open(solStr, "wb") as f:
        pickle.dump(FgPCSolList,f)
endFgPCCoef = time.time()
logger.info("Time for FgPC coefficients: " + \
            '{:.2f}'.format(endFgPCCoef - startFgPCCoef) + " s")

    
# ----------------
# --- Sampling ---
# ----------------

# set strings in order to check if already calculated
strFolder = myResultPath +  "2_calculatedData/variableSampling/"
strSampleBeg = strFolder + tag + "_" + configFgPC.distStr + "_"
strSampleEnd = "_" + '{:.1f}'.format(configFgPC.low) + "_" + '{:.1f}'.format(configFgPC.high) + ".pkl"
sampleStr = strSampleBeg + '{:.0e}'.format(configFgPC.sampleNr) + strSampleEnd

mySamplingModel = samplingModel(configFgPC, sampleStr, logger)
samplesStr = myResultPath + "2_calculatedData/solutionSampling/" + tag + "_H" + str(configFgPC.H) \
    + "_gPC" + str(configFgPC.ngPC) + "_" + configFgPC.distStr + "_" \
        + '{:.0f}'.format(configFgPC.low) + "_" \
            + '{:.0f}'.format(configFgPC.high) + "_s"\
                + '{:.0e}'.format(mySamplingModel.sampleNr)
samplingStr = samplesStr + "_sampling.pkl"
stochasticStr = samplesStr + "_stochastics.pkl"

if os.path.exists(samplingStr):
    with open(samplingStr, "rb") as f:
        samplings = pickle.load(f)

    FgPCResultList = samplings[0]
    mcResultList = samplings[1]

else:

    # The samplingOfModels method calculates the Fourier coefficients of the
    # FgPC solution and the Monte Carlo solution for all given samples
    FgPCResultList, mcResultList = \
        mySamplingModel.samplingOfModels(FgPCSolList,
                                         myBetaCellFgPC,
                                         tagList,
                                         configFgPC,
                                         configInt,
                                         BetaCellInitModel(1),
                                         myResultPath,
                                         systemStr,
                                         logger = logger)

    # --- save results ---
    with open(samplingStr, "wb") as f:
        pickle.dump([FgPCResultList, mcResultList], f)

samplingTimes = [0, np.pi/2, np.pi, 4, 3/2*np.pi, 5.6]
nEvalPt =configFgPC.nPt
timeVec = np.linspace(0, 2*np.pi, nEvalPt)
samplingTimeIdxList = [np.argmin(np.abs(timeVec - timeTarget)) for timeTarget in samplingTimes]

# calculate the stochastics of the FgPC and Monte Carlo solutions
# they contain the mean value and the 95 % sampling interval over
# one oscillation period. The difference between the FgPC and Monte Carlo
# solutions over one period is also calculated. Additionally, the complex
# marginal distribution for given time points is calculated.
if os.path.exists(stochasticStr):
    with open(stochasticStr, "rb") as f:
        stochastics = pickle.load(f)

    FgPCStochasticList = stochastics[0]
    mcStochasticList = stochastics[1]
    diffStochasticList = stochastics[2]
    FgPCsamplingList = stochastics[3]
    mcsamplingList = stochastics[4]

else:

    FgPCStochasticList, mcStochasticList, diffStochasticList, \
        FgPCsamplingList, mcsamplingList = \
                mySamplingModel.getStochastics(myBetaCellFgPC,
                                            FgPCResultList,
                                            mcResultList,
                                            configFgPC.nPt,
                                            len(myBetaCellFgPC.harmonics),
                                            samplingTimeIdxList,
                                            confCoverage)

    with open(stochasticStr, "wb") as f:
        pickle.dump([FgPCStochasticList, 
                     mcStochasticList, 
                     diffStochasticList,
                     FgPCsamplingList,
                     mcsamplingList], f)
        
# -------------------
# --- Convergence ---
# -------------------
if convergenceStudy:
    rmseSamples = 1e4
    convStr = myResultPath + "2_calculatedData/convSampling/" + tag + "_converg_H" + str(configFgPC.H) \
                + "_gPC" + str(configFgPC.ngPC) + "_" + configFgPC.distStr + "_" \
                    + '{:.1f}'.format(configFgPC.low) + "_" \
                        + '{:.1f}'.format(configFgPC.high) + "_s"\
                            + '{:.0e}'.format(rmseSamples)
    if os.path.exists(convStr):
        with open(convStr, "rb") as f:
            convergData = pickle.load(f)
            convergTensor = convergData[0]
            errorMat = convergData[1]

    else:

        # calculates all needed solutions for the convergence study
        # the given H and N of the .yaml file are the maximum values.
        solutionMatrix = mySolutionObj.calFgPCSolutionMatrix(myBetaCellFgPC,
                                                            BetaCellInitModel(1),
                                                            [solStr1, solStr2])
        
        sampleStr = strSampleBeg + '{:.0e}'.format(rmseSamples) + strSampleEnd
        
        myConvergenceModel = convergenceFgPC(solutionMatrix, 
                                            len(varList),
                                            configFgPC.distStr,
                                            configFgPC.low,
                                            configFgPC.high,
                                            sampleStr,
                                            rmseSamples,
                                            forced = False,
                                            logger=logger)

        # calcualtes the convergens map for each variable (convergTensor) and 
        # the sum over all variables (errorMat)
        convergTensor, errorMat = myConvergenceModel.calcConvergenceMap(copy.deepcopy(myBetaCellFgPC))

        with open(convStr, "wb") as f:
            pickle.dump([convergTensor, errorMat], f)

# ----------------
# --- Plotting ---
# ----------------
plotter = Plotter(save_path = result_folder, stylesheet= "paper", save_format= "png", open_saved_plot=False)

myPlottingRoutine = plottingRoutine()

if convergenceStudy:
    # plots the convergence map of the FgPC solution, either for each variable
    # or the total convergence map over all variables

    # myPlottingRoutine.plotConvergenceMap(convergTensor,
    #                                     varList,
    #                                     plotter=plotter)
    
    myPlottingRoutine.plotTotalConvergenceMap(errorMat,
                                              tag,
                                              plotter=plotter,
                                              figSize=(12,7.5))

# plots the magnitude of the FgPC coefficients als a color map
# with H as the x-axis and the gPC order as the y-axis
myPlottingRoutine.plotCoeffMap(FgPCSolList[0][0],
                               configFgPC.H,
                               configFgPC.ngPC,
                               varNr=3,
                               varList=varList,
                               forced=False,
                               plotter=plotter,
                               figSize=(12,7.5))


velPlotBool = False
diffPlotBool = True
xlabelCell = "normalized time"
xticksList = [
    [0, np.pi/2 ,np.pi, 3/2*np.pi, np.pi*2],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3/2 \pi$", r"$2\pi$"]
]
distrList = [sampleStr,"ATP"]
varListScale = [r"$V$ in mV", r"$n$ in [-]", r"$Ca$ in mM"]
confCoverageStr = str(int(confCoverage*100))
figSize = (7,5)

for tagStr, solIdx in zip(tagList, range(len(tagList))):
        
    # get data
    curPosFgPCResultList = FgPCsamplingList[solIdx]
    curPosmcResultList = mcsamplingList[solIdx]

    curFgPCStochasticsList = FgPCStochasticList[solIdx]
    curmcStochasticsList = mcStochasticList[solIdx]
    curdiffStochasticsList = diffStochasticList[solIdx]

    for varIdx in range(len(varList)):

        idxTime = 900

        curPosVarFgPCResult = curPosFgPCResultList[varIdx]
        curPosVarmcResult = curPosmcResultList[varIdx]
        curFgPCStochastics = curFgPCStochasticsList[varIdx]
        curmcStochastics = curmcStochasticsList[varIdx]
        curdiffStochastics = curdiffStochasticsList[varIdx]
        
        # plots the position and, if selected, velocity of the FgPC solution,
        # difference of the FgPC and Monte Carlo solutions, 
        # as well as the marginal distribution
        # of the FgPC solution at the given time points
        myPlottingRoutine.plottingResults(curPosVarFgPCResult,
                                        curPosVarmcResult,
                                        curFgPCStochastics,
                                        curdiffStochastics,
                                        timeVec,
                                        idxTime,
                                        samplingTimes,
                                        samplingTimeIdxList,
                                        confCoverageStr,
                                        plotter,
                                        varListScale[varIdx],
                                        figSize = figSize,
                                        xlabel = xlabelCell,
                                        xticksList = xticksList,
                                        velPlot = velPlotBool,
                                        diffPlot = diffPlotBool,
                                        sampleStr = distrList)
        
        velPlotBool = False
        diffPlotBool = False
        distrList = None

    # plots the phase plots of the FgPC solution
    myPlottingRoutine.getPhasePlots([curFgPCStochasticsList[0][0],
                                    curFgPCStochasticsList[0][2],
                                    curFgPCStochasticsList[0][3]],
                                    [curFgPCStochasticsList[1][0],
                                    curFgPCStochasticsList[1][2],
                                    curFgPCStochasticsList[1][3]],
                                    varListScale[0], varListScale[1], idxTime, 
                                    confCoverageStr, plotter,
                                    timeVec,
                                    figSize = (13.5,9), xlabel = xlabelCell,
                                    xticksList = xticksList,
                                    )
    
    # plots the histogram of the base frequency of the FgPC solution
    omegaPlots = myPlottingRoutine.histoPlots(curPosFgPCResultList[-1][0],
                                              curPosmcResultList[-1],
                                              [curPosFgPCResultList[-1][1],
                                               curPosFgPCResultList[-1][2],
                                               curPosFgPCResultList[-1][3]],
                                               "base frequency")
    
    plotter.plot(omegaPlots, 
                 filename = "omegaHisto",
                 fig_size = figSize)
