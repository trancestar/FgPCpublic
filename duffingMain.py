r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    This is the main filefor the FgPC model of the Duffing system.

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

from duffingFgPC.utils.argparser import get_config_Integration, get_config_FgPC
from duffingFgPC.models.duffingFgPC import duffingFgPC
from duffingFgPC.models.duffingInitialGuess import DuffingInitModel

# ---------------------
# --- Setup section ---
# ---------------------


# 0: small solution,
# 1: large solution, 
# 2: unstable solution, 
# 3: all solutions
systemVal = 1

systemStr = "Duffing"
deflationBool = False

convergenceStudy = True
confCoverage = 0.95

# result folder
if systemVal==1:
    tag = "large"
elif systemVal==2:
    tag = "unstable"
elif systemVal==3:
    tag = "deflation"
    deflationBool = True
else:
    tag = "small"

if systemVal == 3:
    tagList = [ "unstable", "small", "large"]
else:
    tagList = [tag]

# set path where to store results
myResultPath = os.path.realpath(__file__) # use this when storing in same folder

saveResultsPath = myResultPath + "3_results/"

result_folder = save_script(saveResultsPath,
                            os.path.realpath(__file__),
                            systemStr,tag, 
                            max_daily_folders = 7, 
                            max_res_folders = 7)

# logger
logger = init_logger(result_folder, "logfile")

# Get user input from: cmd line > config file > defaults
configFgPC = get_config_FgPC("./duffingFgPC/duffingFgPC.yaml", result_folder, logger)
configInt = get_config_Integration("./duffingFgPC/duffingInt.yaml", result_folder, logger)

distributionStr = configFgPC.distStr + "_" + '{:.1f}'.format(configFgPC.low) + "_" \
        + '{:.1f}'.format(configFgPC.high)
solStr1 = "0_solutionData/" + tag + "_H"
solStr2 = "_" + distributionStr + ".pkl"
solStr = solStr1 + "{:02}".format(configFgPC.H) + "_gPC" + "{:02}".format(configFgPC.ngPC) \
    + solStr2

# ---------------------
# --- FgPC section ---
# ---------------------

if deflationBool:
    exptSol = 3
else:
    exptSol = 1

# The FgPC objects handles the calculation of the FgPC coefficients
mySolutionObj = FgPCsolution(configFgPC, 
                             configInt,
                             myResultPath,
                             systemStr,
                             distributionStr,
                             tag,
                             varNr = 1, 
                             forced = True,
                             exptSol = exptSol,
                             logger = logger,
                             deflationBool = deflationBool)

startFgPCCoef = time.time()
if os.path.exists(solStr):
    with open(solStr, "rb") as f:
        FgPCSolList = pickle.load(f)
    
    myDuffingFgPC = duffingFgPC(configFgPC,
                                configInt,
                                deflation = deflationBool,
                                logger = logger)
    
else:

    myDuffingFgPC = duffingFgPC(configFgPC,
                                configInt,
                                deflation = deflationBool,
                                logger = logger)
    
    # calculates the FgPC coefficients, if deflation is set to True, 
    # all possible solutions are calcualted.
    FgPCSolList = mySolutionObj.calFgPCSolution(myDuffingFgPC, 
                                                DuffingInitModel(),
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
strFolder = myResultPath + "2_calculatedData/variableSampling/"
strSampleBeg = strFolder + tag + "_" + configFgPC.distStr + "_"
strSampleEnd = "_" + '{:.1f}'.format(configFgPC.low) + "_" + '{:.1f}'.format(configFgPC.high) + ".pkl"
sampleStr = strSampleBeg + '{:.0e}'.format(configFgPC.sampleNr) + strSampleEnd

mySamplingModel = samplingModel(configFgPC, sampleStr, logger)

samplesStr = myResultPath + "2_calculatedData/solutionSampling/" + tag + "_H" + str(configFgPC.H) \
    + "_gPC" + str(configFgPC.ngPC) + "_" + configFgPC.distStr + "_" \
        + '{:.1f}'.format(configFgPC.low) + "_" \
            + '{:.1f}'.format(configFgPC.high) + "_s"\
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
                                         myDuffingFgPC,
                                         tagList,
                                         configFgPC,
                                         configInt,
                                         DuffingInitModel(),
                                         myResultPath,
                                         systemStr,
                                         forced = True,
                                         logger = logger)

    # --- save results ---
    with open(samplingStr, "wb") as f:
        pickle.dump([FgPCResultList, mcResultList], f)

samplingTimes = [0, 2]
timeVec = np.linspace(0, 2*np.pi/configInt.omega, configFgPC.nPt)
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
                mySamplingModel.getStochastics(myDuffingFgPC,
                                            FgPCResultList,
                                            mcResultList,
                                            configFgPC.nPt,
                                            len(myDuffingFgPC.harmonics),
                                            samplingTimeIdxList,
                                            confCoverage,
                                            forced = True)

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
        solutionMatrix = mySolutionObj.calFgPCSolutionMatrix(myDuffingFgPC,
                                                            DuffingInitModel(),
                                                            [solStr1, solStr2])
        
        sampleStr = strSampleBeg + '{:.0e}'.format(rmseSamples) + strSampleEnd
        
        myConvergenceModel = convergenceFgPC(solutionMatrix, 
                                            1,
                                            configFgPC.distStr,
                                            configFgPC.low,
                                            configFgPC.high,
                                            sampleStr,
                                            rmseSamples,
                                            logger=logger)

        # calcualtes the convergens map for each variable (convergTensor) and 
        # the sum over all variables (errorMat)
        convergTensor,errorMat = myConvergenceModel.calcConvergenceMap(copy.deepcopy(myDuffingFgPC))

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
    #                                     varList=['x'],
    #                                     plotter=plotter)
    myPlottingRoutine.plotTotalConvergenceMap(errorMat,
                                              tag,
                                              plotter=plotter,
                                              figSize=(12,7.5))

# plots the magnitude of the FgPC coefficients als a color map
# with H as the x-axis and the gPC order as the y-axis
myPlottingRoutine.plotCoeffMap(FgPCSolList[0],
                               configFgPC.H,
                               configFgPC.ngPC,
                               varNr=1,
                               varList=['x'],
                               forced=True,
                               plotter=plotter,
                               figSize=(12,7.5))

distrList = [sampleStr, r"$\alpha$"]
confCoverageStr = str(int(confCoverage*100))
varIdx = 0
for tagStr, solIdx in zip(tagList, range(len(tagList))):

    if tagStr == "small":
        idxTime = 0
    elif tagStr == "unstable":
        idxTime = 450
    else:
        idxTime = 900
    
    # assign data
    curPosFgPCResult = FgPCsamplingList[solIdx][varIdx]
    curPosmcResult = mcsamplingList[solIdx][varIdx]

    curFgPCStochastics = FgPCStochasticList[solIdx][varIdx]
    curmcStochastics = mcStochasticList[solIdx][varIdx]
    curdiffStochastics = diffStochasticList[solIdx][varIdx]

    # plots the position and, if selected, velocity of the FgPC solution,
    # difference of the FgPC and Monte Carlo solutions, 
    # as well as the marginal distribution
    # of the FgPC solution at the given time points
    myPlottingRoutine.plottingResults(curPosFgPCResult,
                                      curPosmcResult,
                                      curFgPCStochastics,
                                      curdiffStochastics,
                                      timeVec,
                                      idxTime,
                                      samplingTimes,
                                      samplingTimeIdxList,
                                      confCoverageStr,
                                      plotter,
                                      r'$x$',
                                      velPlot = True,
                                      diffPlot = True,
                                      sampleStr = distrList,
                                      figSize = (7,5))
    
    distrList = None

    # plots the phase plots of the FgPC solution
    myPlottingRoutine.getPhasePlots([curFgPCStochastics[0],
                                     curFgPCStochastics[2],
                                     curFgPCStochastics[3]],
                                    [curFgPCStochastics[1],
                                     curFgPCStochastics[4],
                                     curFgPCStochastics[5]],
                                    r'$x$', r'$v$', idxTime, 
                                    confCoverageStr, plotter,
                                    timeVec,
                                    figSize = (13.5,9)
                                    )
