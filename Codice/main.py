import importlib

import libAnalyseNetwork as libAN
importlib.reload(libAN)
import libExtractData as libED
importlib.reload(libED)
import libMASKineticTheory as libKT
importlib.reload(libKT)
import libGUI
importlib.reload(libGUI)

import numpy as np


### Graphic User Interface ###
dicPrm = libGUI.GUI() # Parameters


if dicPrm['runState']:

    ### Data extraction ###
    if dicPrm['extraction']:
        libED.ExtractAdjacencyMatrices()


    ### Matrices extraction ###
    A, W = libED.ReadAdjacencyMatrices(dicPrm['regSelected'])
    dicPrm['numberCities'] = A.shape[0] # Number of nodes


    ### Network analysis ###
    if dicPrm['analysis']:
        # Degrees
        di = np.sum(A,axis=0) # Vectors of degrees
        dk, Nk = np.unique(di,return_counts=True)
        # Unique degrees and corresponding frequencies
        # Pk = counts/N

        # [Nonzero] Weights
        wi = W[W>0]
        wk, wNk = np.unique(wi,return_counts=True)
        # Unique weights and corresponding frequencies

        # Strenghts
        si = np.sum(W,axis=0)
        sk, sNk = np.unique(si,return_counts=True)
        # Unique strenghts and corresponding frequencies

        libAN.DegreeDistributionFig(di)
        libAN.WeightDistributionFig(wi)
        libAN.StrengthDistributionFig(si)

        libAN.BetweennessCentralityFig(A,di)
        libAN.StrengthVsDegreeFig(si,di,dk,Nk)

        Ck = libAN.AClusteringCoefficientFig(A,di,dk,Nk)
        libAN.WClusteringCoefficientFig(A,W,si,di,dk,Nk,Ck)

        knn = libAN.AAssortativityFig(A,dk)
        libAN.WAssortativityFig(W,dk,knn)

        # plt.show()
        # mpld3.show()


    ### Kinetic simulation ###
    # stateCities = libKT.MonteCarlo(
    #     A,
    #     dicPrm['totalPop'],
    #     dicPrm['numberCities'],
    #     dicPrm['attractivity'],
    #     dicPrm['convincibility'],
    #     dicPrm['deviation'],
    #     dicPrm['stepNumber'],
    #     dicPrm['timeStep']
    # )

    # libKT.CityDistributionFig(
    #     stateCities.vtxState,
    #     dicPrm['numberCities']
    # )
    # libKT.CityAverageFig(
    #     stateCities.avgState,
    #     dicPrm['stepNumber'],
    #     dicPrm['timeStep']
    # )