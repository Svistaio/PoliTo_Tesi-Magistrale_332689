
import importlib

import libAnalyseNetwork as libAN
importlib.reload(libAN)
import libExtractData as libED
importlib.reload(libED)
import libMASKineticTheory as libKT
importlib.reload(libKT)
import libGUI
importlib.reload(libGUI)


### Graphic User Interface ###
dicPrm = libGUI.GUI() # Parameters


if dicPrm['runState']:

    ### Data extraction ###
    if dicPrm['extraction']:
        libED.ExtractAdjacencyMatrices()


    ### Matrices extraction ###
    dicReg = libED.ReadAdjacencyMatrices(dicPrm['regSelected'])


    ### Network analysis ###
    if dicPrm['analysis']:
        clsNA = libAN.NetworkAnalysis(dicReg)

        clsNA.DegreeDistributionFig()
        clsNA.WeightDistributionFig()
        clsNA.StrengthDistributionFig()

        clsNA.BetweennessCentralityFig()
        clsNA.StrengthVsDegreeFig()

        clsNA.AClusteringCoefficientFig()
        clsNA.WClusteringCoefficientFig()

        clsNA.AAssortativityFig()
        clsNA.WAssortativityFig()

        # clsNA.ShowFig()


    ### Kinetic simulation ###
    clsKS = libKT.KineticSimulation(dicPrm,dicReg)

    clsKS.MonteCarloAlgorithm()
    clsKS.SizeDistrFitsFig()
    clsKS.SizeAverageFig()
    clsKS.SizeVsDegreeFig()
    # clsKS.SizeDistrEvolutionFig()
