import numpy as np

import importlib
from tqdm import tqdm

import libFigures as libF
importlib.reload(libF)

import matplotlib.pyplot as plt
libF.SetTextStyle()


### Main functions ###

def MonteCarlo(
    A,totP,Nn,
    l,a,s,
    Nt,dt
):
    stateCities = NetworkState(totP,Nn,A,Nt,l,a,s)

    for nt in tqdm(range(Nt),desc="Updating states"):
        stateCities.updateState(dt,nt)

    return stateCities 

def CityDistributionFig(cs,Nn):
    fig, ax = plt.subplots(1,2,figsize=(15,7))
    dicData = libF.CreateDicData(2)

    labels = ["Ext.","Apx."]
    colours = ["blue","red"]

    for i,type in enumerate(cs):

        ### Lognormal fit ###
        libF.CreateHistogramPlot(
            cs[type],30,dicData['fig1'],
            l=f"{labels[i]} histogram",
            clr=colours[i],alfa=0.5,
            idx=i+1,ax=ax[0]
        ) # Histogram plot

        libF.CreateLognormalFitPlot(
            cs[type],dicData['fig1'],
            lAvr=fr"{labels[i]} mean value $\langle k\rangle$",
            lFit=f"{labels[i]} lognormal fit (ML)",
            clrAvr=colours[i],clrFit=colours[i],
            idx=i+1,ax=ax[0]
        ) # Fit plot


        ### Power law fit ###
        b = libF.CreateParetoFitPlot(
            cs[type],dicData['fig2'],
            lSct=f"{labels[i]} empirical CCDF",
            lFit=fr"{labels[i]} pareto fit",
            clrSct=colours[i],clrFit=colours[i],
            idx=i+1,ax=ax[1]
        )

        fig.text(
            .5,.925+i*.05,
            fr"{labels[i]}: $\quad s_{{min}}={np.min(cs[type]):.2f}\qquad$"
            fr"$s_{{max}}={np.max(cs[type]):.2f}\qquad$"
            fr"$\langle s\rangle ={np.mean(cs[type]):.2f}\qquad$"
            fr"$s_{{\Sigma}}={np.sum(cs[type]):.2f}\qquad$"
            fr"$\alpha={b:.2f}$",
            ha="center",
            # fontsize=10,
            color="black"
        )

    fig.text(.1,.95,fr"$N={Nn}\qquad$")


    # Style
    libF.SetFigStyle(
        r"$s$",r"$P(s)$",
        yNotation="sci", # ,xNotation="sci"
        ax=ax[0],data=dicData['fig1']
    )

    libF.SetFigStyle(
        r"$s$",r"$P(s)$",
        xScale="log",yScale="log",
        ax=ax[1],data=dicData['fig2']
    )


    libF.CentreFig()
    libF.SaveFig('CitySizeDistribution','KS',dicData)

def CityAverageFig(ca,Nt,dt):
    fig = plt.figure()
    dicData = libF.CreateDicData(1)

    labels = ["Ext.","Apx."]
    colours = ["blue","red"]
    timeInterval = np.arange(Nt+1)*dt

    for i,type in enumerate(ca):
        libF.CreateFunctionPlot(
            timeInterval,ca[type],dicData['fig'],
            l=rf"{labels[i]} city size average $\langle s\rangle$",
            clr=colours[i],idx=i+1
        )

    libF.CentreFig()
    libF.SetFigStyle(
        r"$t$",r"$\langle s\rangle$",
        data=dicData['fig']
    )
    libF.SaveFig('AverageCitySize','KS',dicData)


### Auxiliary functions ###

class NetworkState:
    def __init__(self,totP,Nn,A,Nt,l,a,sigma):
        # Uniform initial state for all vertices
        keys = ["ext","apx"]

        self.vtxState   = {
            keys[0]:np.ones((Nn,1),dtype=float)*(totP/Nn),
            keys[1]:np.ones((Nn,1),dtype=float)*(totP/Nn),
        }
        self.avgState    = {
            keys[0]:np.zeros((Nt+1,1),dtype=float),
            keys[1]:np.zeros((Nt+1,1),dtype=float)
        }
        for k in self.avgState:
            self.avgState[k][0] = totP/Nn 

        # Exact Adjacency matrix
        self.A  = A
        
        # Approximated adjacency matrix
        Mn = np.sum(A)
        wO = np.sum(A,axis=1,keepdims=True)
        wI = np.sum(A,axis=0,keepdims=True)
        self.R  = wO@wI/Mn
        # In my case wO=wI but it's better to define them in the most general way

        self.M = {keys[0]:self.A, keys[1]:self.R}

        def E(si,sr): return NonLinearEmigration(si,sr,l,a)
        self.E = E

        def ga(E): return StochasticFluctuations(sigma,E)
        self.ga = ga

        self.Nn = Nn # Number of nodes

    def updateState(self,dt,nt):
        oldState = self.vtxState
        newState = {k: v.copy() for k, v in self.vtxState.items()}
 
        P = np.random.permutation(self.Nn)

        halfN = int(np.floor(self.Nn/2))
        p1 = P[:halfN]; p2 = P[halfN+1:]

        for i in range(halfN):
            for k in self.M:
                # It's assumed node p1(i) is the interacting node while node p2(i) is the receiving one

                p = float(np.clip(self.M[k][p1[i],p2[i]]*dt,0,1))
                theta = np.random.binomial(1,p)
                si = oldState[k][p1[i]]; sr = oldState[k][p2[i]]

                E = self.E(si,sr); ga = self.ga(E)
                newState[k][p1[i]] = si*(1-theta)+theta*si*(1-E+ga) 
                newState[k][p2[i]] = sr*(1-theta)+theta*(sr+si*E)

        self.vtxState = newState
        self.avgState["ext"][nt+1] = np.mean(newState["ext"])
        self.avgState["apx"][nt+1] = np.mean(newState["apx"])

def NonLinearEmigration(
        si, # Interacting city size
        sr, # Receiving city size
        l,  # Maximum emigration rate
        a   # Emigration intensity
    ):
    if si != 0:
        rs = sr/si               # Relative population ratio
        ef = l*(rs**a)/(1+rs**a) # Actual emigration rate
    else:
        ef = 0
    return ef

def StochasticFluctuations(sigma,E):
    alpha = ((1-E)**2)/(sigma**2)
    theta = (sigma**2)/(1-E)

    if alpha<=1:
        raise ValueError(
            "α must be >1 to have a non-degenerate gamma distribution,"
            " and thus always admissible fluctuations"
        )

    ga = np.random.gamma(alpha,theta)
    return ga+E-1


### Discarded code ###

#region Old implementation for fluctuations with a [forcefully] resampled Gaussian until the value picked ensures the post-interaction population is positive
    # def StochasticFluctuations(sigma,E):
    #     while True:
    #         mu = np.random.normal(0,sigma,size=1)
    #         if mu>E-1: #and mu<E:
    #             break
    #     # The conditions «mu>E-1» and «mu<E» are necessary to have the total emigration rage 1-E+μ between 0 and 1
    #     return mu
#endregion
