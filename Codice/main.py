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


### GUI ###
prm = libGUI.GUI() # Parameters


if prm['runState']:

    ### Network section ###

    if prm['extData']:
        zipPath = "../Dati/matriciPendolarismo1991.zip"
        libED.ExtractAdjacencyMatrices(zipPath)

    A, W = libED.ReadAdjacencyMatrices(
        "../Dati/AdjacencyMatricesIt91.zip",
        "20AdjacencyMatrixSardegna.txt",
        "20WeightedAdjacencyMatrixSardegna.txt"
    )
    Nn = A.shape[0] # Number of nodes

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

    if prm['extData']:
        libAN.DegreeDistributionFig(di,Nn)
        libAN.WeightDistributionFig(wi)
        libAN.StrengthDistributionFig(si)

        libAN.BetweennessCentralityFig(A,di)
        libAN.StrengthFromDegreeFig(si,di,dk,Nk)

        Ck = libAN.ClusteringCoefficientFig(A,di,dk,Nk)
        libAN.WeightedClusteringCoefficientFig(A,W,si,di,dk,Nk,Ck)

        knn = libAN.AssortativityFig(A,dk)
        libAN.WeightedAssortativityFig(W,dk,knn)

        # plt.show()
        # mpld3.show()


    ### Kinetic section ###

    stateCities = libKT.MonteCarlo(
        prm['sMax'],Nn,A,
        prm['Nt'],prm['dt'],
        prm['l'],prm['a'],prm['sigma']
    )
    libKT.CityDistributionFig(stateCities.vtxState,Nn)
    libKT.CityAverageFig(stateCities.avgState,prm['Nt'],prm['dt'])