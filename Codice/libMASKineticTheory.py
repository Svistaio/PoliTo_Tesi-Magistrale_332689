import numpy as np
from scipy.stats import lognorm, linregress, pareto

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

    csAvr = np.mean(cs); csSum = np.sum(cs)
    csMin = np.min(cs); csMax = np.max(cs)


    ### Lognormal fit ###

    # Histogram plot
    hgPlot = ax[0].hist(cs,
        # bins='auto',
        bins=30,
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label="Histogram"
    )

    # mplcursors.cursor(hgPlot[2],hover=False).connect(
    #     "add",lambda sel: sel.annotation.set_text(
    #         "$P^{bin}(k)$="+f"{hgPlot[0][sel.index]:.3f}"
    #     )
    # )

    shape, loc, scale = lognorm.fit(cs,floc=0)
    ax[0].plot(
        [csAvr,csAvr],[0,lognorm.pdf(csAvr,shape,loc=loc,scale=scale)],
        label=r"Mean value $\langle k\rangle$",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    x = np.linspace(0,np.max(cs),500)
    fPlot = ax[0].plot(
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
    libAN.SetPlotStyle(
        r"$s$",r"$P(s)$",ax=ax[0],
        yNotation="sci" # ,xNotation="sci"
    )


    ### Power law fit ###

    # Histogram plot
    xmin = np.percentile(cs,75)
    xtail = cs[cs >= xmin]
    b, loc, scale = pareto.fit(xtail,floc=0,fscale=xmin)  # b≈alpha

    # Empirical CCDF on the tail
    xs = np.sort(xtail) # Ascending values
    n = xs.size
    ccdfEmp = 1.0 - np.arange(1,n+1,dtype=float)/n # P(X≥x)

    # Model CCDF from the fitted Pareto
    ccdfFit = pareto.sf(xs,b,loc=loc,scale=scale) # Survival function

    ax[1].plot(
        xs,ccdfEmp,
        marker="o",
        linewidth=1,
        # linestyle="--",
        linestyle="none",
        label="Empirical CCDF"
    )
    ax[1].plot(
        xs,ccdfFit,
        color='blue',
        label=fr"Pareto fit,$\alpha={b:.2f}$"
    )

    libAN.SetPlotStyle(
        r"$s$",ax=ax[1],
        xScale="log",yScale="log"
    )
    

    # hgPlot = ax[1].hist(cs,
    #     bins=np.logspace(
    #         np.log10(np.min(cs)),
    #         np.log10(np.max(cs)),
    #         40
    #     ),
    #     density=True,
    #     color="gray",
    #     edgecolor="none", # "black"
    #     label="Histogram"
    # )

    # binx = (hgPlot[1][1:]+hgPlot[1][:-1])/2
    # biny = hgPlot[0]

    # # Select the last quarter of city sizes
    # f = 1/4; v = binx>=csMin*(1-f)+csMax*f

    # logBinxV = np.log10(binx[v]); logBinyV = np.log10(biny[v])
    # slope, intercept, _, _, _ = linregress(logBinxV,logBinyV)
    # regression = 10**(intercept+slope*logBinxV)

    # fPlot = ax[1].plot(
    #     binx[v],regression,
    #     label="Regression",
    #     color="blue",
    #     linewidth=1
    # )


    fig.text(
        0.5,0.95,
        fr"$N$="f"{Nn}\\t"
        fr"$s_{{min}}$={csMin:.2f}\\t"
        fr"$s_{{max}}$={csMax:.2f}\\t"
        fr"$\langle s\rangle $={csAvr:.2f}\\t"
        fr"$s_{{Sum}}=${csSum:.2f}\\t"
        fr"$\alpha=${b:.2f}",
        ha="center",
        # fontsize=10,
        color="black"
    )

    libAN.CentrePlot()
    libAN.SaveFig(fig,"CitySizeDistributionSardegna")


def CityAverageFig(ca,Nt,dt):
    fig = plt.figure()

    timeInterval = np.arange(Nt+1)*dt

    plt.plot(
        timeInterval,ca,
        # label=r"etichetta",
        color="black",
        linewidth=1,
    )

    libAN.SetPlotStyle(r"$t$",r"$\langle s\rangle$")
    libAN.CentrePlot()
    libAN.SaveFig(fig,"AverageCitySizeSardegna")



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
            newState[p1[i]] = si*(1-theta)+theta*si*(1-E) # +mu
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