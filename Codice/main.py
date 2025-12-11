
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
clsPrm = libGUI.GUI() # Parameters


if clsPrm.simFlag:

    ### Data extraction ###
    if clsPrm.extraction: libED.ExtractAdjacencyMatrices()


    ### Matrices extraction ###
    clsReg = libED.ReadAdjacencyMatrices(clsPrm.regSelected)


    ### Network analysis ###
    if clsPrm.analysis:
        clsNA = libAN.NetworkAnalysis(clsPrm,clsReg)

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
    clsKS = libKT.KineticSimulation(clsPrm,clsReg)

    clsKS.MonteCarloAlgorithm()
    clsKS.SizeDistrFittingsFig()
    clsKS.AverageSizeFig()
    clsKS.SizeVsDegreeFig()
    clsKS.SizeDistrEvolutionFig()
    clsKS.SizeEvolutionsFig()
