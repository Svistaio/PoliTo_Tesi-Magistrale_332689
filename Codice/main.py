import importlib

import libAnalyseNetwork as libAN
importlib.reload(libAN)
import libExtractData as libED
importlib.reload(libED)
import libMASKineticTheory as libKT
importlib.reload(libKT)

import numpy as np



### Network section ###

# zipPath = "../Dati/matriciPendolarismo1991.zip"
# libED.ExtractAdjacencyMatrices(zipPath)

A, W = libED.ReadAdjacencyMatrices(
    "../Dati/AdjacencyMatricesIt91.zip",
    "20AdjacencyMatrixSardegna.txt",
    "20WeightedAdjacencyMatrixSardegna.txt"
)
N = A.shape[0]

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

# libAN.DegreeDistributionFig(di)
# libAN.WeightDistributionFig(wi)
# libAN.StrengthDistributionFig(si)

# libAN.BetweennessCentralityFig(A,di)
# libAN.StrengthFromDegreeFig(si,di,dk,Nk)

# Ck = libAN.ClusteringCoefficientFig(A,di,dk,Nk)
# libAN.WeightedClusteringCoefficientFig(A,W,si,di,dk,Nk,Ck)

# knn = libAN.AssortativityFig(A,dk)
# libAN.WeightedAssortativityFig(W,dk,knn)

# plt.show()
# mpld3.show()



### Kinetic section ###

# popSard = 1600000 # Sardinia population in 1991

# dt = 1e-2
# Nt = 2e4

# libKT.Initialize(popSard)
# libKT.MonteCarlo()