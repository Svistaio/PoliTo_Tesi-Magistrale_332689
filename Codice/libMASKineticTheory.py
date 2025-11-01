import numpy as np
from scipy.stats import lognorm, pareto

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
    fig, ax = plt.subplots(1,2,figsize=(15,7))

    labels = ["Ext.","Apx."]
    colours = ["blue","red"]

    csAvr = np.array((2,1)); csSum = np.array((2,1))
    csMin = np.array((2,1)); csMax = np.array((2,1))

    for i,k in enumerate(cs):
        csAvr[i] = np.mean(cs[k]); csSum[i] = np.sum(cs[k])
        csMin[i] = np.min(cs[k]); csMax[i] = np.max(cs[k])


        ### Lognormal fit ###

        # Histogram plot
        hgPlot = ax[0].hist(
            cs[k],
            # bins='auto',
            bins=30,
            density=True,
            color=colours[i],
            edgecolor="none", # "black"
            label=f"{labels[i]} histogram",
            alpha=0.5
        )

        shape, loc, scale = lognorm.fit(cs[k],floc=0)
        ax[0].plot(
            [csAvr[i],csAvr[i]],
            [0,lognorm.pdf(csAvr[i],shape,loc=loc,scale=scale)],
            label=fr"{labels[i]} mean value $\langle k\rangle$",
            color=colours[i],
            linewidth=1,
            linestyle="--"
        )

        s = np.linspace(0,np.max(cs[k]),500)
        fPlot = ax[0].plot(
            s,lognorm.pdf(s,shape,loc=loc,scale=scale),
            label=f"{labels[i]} lognormal fit (ML)", # Maximum likelyhood
            color=colours[i],
            linewidth=1
        ) # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»


        ### Power law fit ###

        csQuarter = np.percentile(cs[k],75)
        csTail = cs[k][cs[k] >= csQuarter]
        b, loc, scale = pareto.fit(csTail,floc=0,fscale=csQuarter) # b≈alpha

        # # Select the last quarter of city sizes
        # f = 1/4; v = binx>=csMin*(1-f)+csMax*f
        # binx = (hgPlot[1][1:]+hgPlot[1][:-1])/2
        # biny = hgPlot[0]

        # Empirical CCDF on the tail
        csSort = np.sort(csTail) # Ascending values
        n = csSort.size
        ccdfEmp = 1 - np.arange(1,n+1,dtype=float)/n # P(X≥x)

        # Model CCDF from the fitted Pareto
        ccdfFit = pareto.sf(csSort,b,loc=loc,scale=scale) # Survival function

        ax[1].plot(
            csSort,ccdfEmp,
            marker="o",
            color=colours[i],
            linewidth=1,
            # linestyle="--",
            linestyle="none",
            label=f"{labels[i]} empirical CCDF",
            alpha=0.5
        )
        ax[1].plot(
            csSort,ccdfFit,
            color=colours[i],
            label=fr"{labels[i]} pareto fit$"
        )

        fig.text(
            .5,.925+i*.05,
            fr"{labels[i]}: $\quad s_{{min}}={csMin[i]:.2f}\qquad$"
            fr"$s_{{max}}={csMax[i]:.2f}\qquad$"
            fr"$\langle s\rangle ={csAvr[i]:.2f}\qquad$"
            fr"$s_{{\Sigma}}={csSum[i]:.2f}\qquad$"
            fr"$\alpha={b:.2f}$",
            ha="center",
            # fontsize=10,
            color="black"
        )

    fig.text(.1,.95,fr"$N={Nn}\qquad$")

    libAN.SetPlotStyle(
        r"$s$",ax=ax[1],
        xScale="log",yScale="log"
    )

    # Style
    libAN.SetPlotStyle(
        r"$s$",r"$P(s)$",ax=ax[0],
        yNotation="sci" # ,xNotation="sci"
    )

    libAN.CentrePlot()
    libAN.SaveFig(fig,"CitySizeDistributionSardegna")


def CityAverageFig(ca,Nt,dt):
    fig = plt.figure()

    labels = ["Ext.","Apx."]
    colours = ["blue","red"]
    timeInterval = np.arange(Nt+1)*dt

    for i,k in enumerate(ca):
        plt.plot(
            timeInterval,ca[k],
            label=rf"{labels[i]} city size average $\langle s\rangle$",
            color=colours[i],
            linewidth=1,
        )

    libAN.SetPlotStyle(r"$t$",r"$\langle s\rangle$")
    libAN.CentrePlot()
    libAN.SaveFig(fig,"AverageCitySizeSardegna")



### Auxiliary code ###

class networkState:
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

        def mu(E): return StochasticFluctuations(sigma,E)
        self.mu = mu

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

                E = self.E(si,sr); mu = self.mu(E)
                newState[k][p1[i]] = si*(1-theta)+theta*si*(1-E+mu) 
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
        rs = sr/si           # Relative population
        ef = l*(rs**a)/(1+rs**a) # Actual emigration rate
    else:
        ef = 0
    return ef


def StochasticFluctuations(sigma,E):
    while True:
        mu = np.random.normal(0,sigma,size=1)
        if mu>E-1: #and mu<E:
            break
    # The conditions «mu>E-1» and «mu<E» are necessary to have the total emigration rage 1-E+μ between 0 and 1
    return mu