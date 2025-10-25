import importlib

import libAnalyseNetwork as libAN
importlib.reload(libAN)
import libExtractData as libED
importlib.reload(libED)

import numpy as np

# with ZF("../Dati/AdjacencyMatricesIt91.zip") as z:
#     with z.open("20AdjacencyMatrixSardegna.txt","r") as f:
#         A = np.loadtxt( # Decodes binary stream as UTF-8 text
#             tiow(f,encoding="utf-8"),
#             delimiter=",",
#             dtype=int
#         )

# zipPath = "../Dati/matriciPendolarismo1991.zip"
# libED.ExtractData(zipPath)

A, W = libED.ReadAdjacencyMatrices(
    "../Dati/AdjacencyMatricesIt91.zip"
    "20AdjacencyMatrixSardegna.txt",
    "20WeightedAdjacencyMatrixSardegna.txt"
)

# Vectors of degrees
d = np.sum(A, axis=0)
N = len(d) # Number of nodes (normalization factor)

# Unique degrees and corresponding frequencies
k, Nk = np.unique(d,return_counts=True)
# Pk = counts/N

# libAN.DegreeDistributionFig(d,N,k)
# libAN.ClusteringCoefficientFig(A,d,k,Nk)
# libAN.AssortativityFig(A,k)
libAN.BetweennessCentralityFig(A,d)

# plt.show()
# mpld3.show()