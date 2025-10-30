import numpy as np
from scipy.stats import lognorm

import importlib
from tqdm import tqdm

import libAnalyseNetwork as libAN
importlib.reload(libAN)

import matplotlib.pyplot as plt
libAN.SetTextStyle()


### Main code ###

def MonteCarlo(totP,Nn,A,Nt,dt,l,a,sigma):
    stateCities = networkState(totP,Nn,A,Nt,l,a,sigma)

    for nt in tqdm(range(Nt),desc="Updating states"):
        stateCities.updateState(dt,nt)

    return stateCities 


def CityDistributionFig(cs,Nn):
    fig = plt.figure()

    csAvr = np.mean(cs)
    fig.text(
        0.8,0.25,
        r"$N$="+f"{Nn}"+"\n"+\
        r"$s_{min}$="+f"{np.min(cs)}"+"\n"+\
        r"$s_{max}$="+f"{np.max(cs)}"+"\n"+\
        r"$\langle s\rangle $="+f"{csAvr}",
        ha="center",
        # fontsize=10,
        color="black"
    )


    # Histogram plot
    hgPlot = plt.hist(cs,
        # bins=int(np.mean(d)),
        bins=25,
        # bins=60,
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label="Histogram"
    )

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    # mplcursors.cursor(hgPlot[2],hover=False).connect(
    #     "add",lambda sel: sel.annotation.set_text(
    #         "$P^{bin}(k)$="+f"{hgPlot[0][sel.index]:.3f}"
    #     )
    # )

    # SciPy fitting (ML)
    x = np.linspace(0,np.max(cs),500)

    shape, loc, scale = lognorm.fit(cs,floc=0)
    plt.plot(
        [csAvr,csAvr],[0,lognorm.pdf(csAvr,shape,loc=loc,scale=scale)],
        label=r"Mean value $\langle k\rangle$",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    fPlot = plt.plot(
        x,lognorm.pdf(x,shape,loc=loc,scale=scale),
        label="SciPy lognormal fit (ML)", # Maximum likelyhood
        color="blue",
        linewidth=1
    ) # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»

    # mplcursors.cursor(fPlot,hover=False).connect(
    #     "add",lambda sel: sel.annotation.set_text(
    #         f"k={sel.target[0]:.3f}, LN(k)={sel.target[1]:.3f}"
    #     )
    # )

    # Style
    libAN.CentrePlot()
    libAN.SetPlotStyle(r"$s$",r"$P(s)$")
    libAN.SaveFig(fig,"CitySizeDistributionSardegna")


### Auxiliary code ###

class networkState:
    def __init__(self,totP,Nn,A,Nt,l,a,sigma):
        # Uniform initial state for all vertices
        self.verticesState   = np.ones((Nn,1),dtype=float)*(totP/Nn)
        self.averageState    = np.zeros((Nt+1,1),dtype=float)
        self.averageState[0] = totP/Nn

        self.Nn              = Nn # Number of nodes
        self.A               = A  # Adjacency matrix

        def E(si,sr): return NonLinearEmigration(si,sr,l,a)
        self.E = E

        def mu(E): return StochasticFluctuations(sigma,E)
        self.mu = mu

    def updateState(self,dt,nt):
        oldState = self.verticesState
        newState = np.copy(self.verticesState)
        P = np.random.permutation(self.Nn)

        halfN = int(np.floor(self.Nn/2))
        p1 = P[:halfN]; p2 = P[halfN+1:]

        for i in range(halfN):
            p = float(np.clip(self.A[p1[i],p2[i]]*dt,0,1))
            theta = np.random.binomial(1,p)
            si = oldState[p1[i]]; sr = oldState[p2[i]]
            # It's assumed node p1(i) is the interacting node while node p2(i) is the receiving one

            E = self.E(si,sr); mu = self.mu(E)
            newState[p1[i]] = si*(1-theta)+theta*si*(1-E+mu)
            newState[p2[i]] = sr*(1-theta)+theta*(sr+si*E)

        self.verticesState = newState
        self.averageState[nt+1] = np.mean(newState)


def NonLinearEmigration(
        si, # Interacting city size
        sr, # Receiving city size
        l,  # Maximum emigration rate
        a   # Emigration intensity
    ):
    if si != 0:
        rs = sr/si           # Relative population
        ef = l*(rs**a)/(1+rs**a) # Actual emigration rate
    else:
        ef = 0
    return ef


def StochasticFluctuations(sigma,E):
    while True:
        mu = np.random.normal(0,sigma,size=1)
        if mu>E-1 and mu<E:
            break
    # The conditions «mu>E-1» and «mu<E» are necessary to have the total emigration rage 1-E+μ between 0 and 1
    return mu