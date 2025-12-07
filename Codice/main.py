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
prm = libGUI.GUI() # Parameters


if prm['runState']:

    ### Data extraction ###
    if prm['extraction']:
        zipPath = "../Dati/matriciPendolarismo1991.zip"
        libED.ExtractAdjacencyMatrices(zipPath)


    ### Matrices extraction ###
    A, W = libED.ReadAdjacencyMatrices(
        "../Dati/AdjacencyMatricesIt91.zip",
        "20AdjacencyMatrixSardegna.txt",
        "20WeightedAdjacencyMatrixSardegna.txt"
    )
    Nn = A.shape[0] # Number of nodes


    ### Network analysis ###
    if prm['analysis']:
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

        libAN.DegreeDistributionFig(di,Nn)
        libAN.WeightDistributionFig(wi)
        libAN.StrengthDistributionFig(si)

        libAN.BetweennessCentralityFig(A,di)
        libAN.StrengthFromDegreeFig(si,di,dk,Nk)

        Ck = libAN.AClusteringCoefficientFig(A,di,dk,Nk)
        libAN.WClusteringCoefficientFig(A,W,si,di,dk,Nk,Ck)

        knn = libAN.AAssortativityFig(A,dk)
        libAN.WAssortativityFig(W,dk,knn)

        # plt.show()
        # mpld3.show()


    ### Kinetic simulation ###
    stateCities = libKT.MonteCarlo(
        prm['totalPop'],Nn,A,
        prm['stepNumber'],
        prm['timeStep'],
        prm['attractivity'],
        prm['convincibility'],
        prm['deviation']
    )

    libKT.CityDistributionFig(
        stateCities.vtxState,
        Nn
    )
    libKT.CityAverageFig(
        stateCities.avgState,
        prm['stepNumber'],
        prm['timeStep']
    )