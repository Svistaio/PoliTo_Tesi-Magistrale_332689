import numpy as np
from scipy.stats import lognorm, pareto

import importlib
from tqdm import tqdm

import libFigures as libF
importlib.reload(libF)

import matplotlib.pyplot as plt
libF.SetTextStyle()


### Main functions ###

def MonteCarlo(totP,Nn,A,Nt,dt,l,a,sigma):
    stateCities = NetworkState(totP,Nn,A,Nt,l,a,sigma)

    for nt in tqdm(range(Nt),desc="Updating states"):
        stateCities.updateState(dt,nt)

    return stateCities 

def CityDistributionFig(cs,Nn):
    fig, ax = plt.subplots(1,2,figsize=(15,7))
    dicData = {'plots':{'fig1':{},'fig2':{}},'style':{'fig1':{},'fig2':{}}}

    labels = ["Ext.","Apx."]
    colours = ["blue","red"]

    csAvr = np.array((2,1)); csSum = np.array((2,1))
    csMin = np.array((2,1)); csMax = np.array((2,1))

    for i,k in enumerate(cs):
        idx = str(i+1)
        csAvr[i] = np.mean(cs[k]); csSum[i] = np.sum(cs[k])
        csMin[i] = np.min(cs[k]); csMax[i] = np.max(cs[k])


        ### Lognormal fit ###
        # Histogram plot
        x = cs[k]; nBins = 30; l = f"{labels[i]} histogram"
        hgPlot = ax[0].hist(
            x,
            # bins='auto',
            bins=nBins,
            density=True,
            color=colours[i],
            edgecolor="none", # "black"
            label=l,
            alpha=0.5
        )
        dicData['plots']['fig1']['histogramPlot'+idx] = {'t':'histogram','x':x,'b':nBins,'l':l}

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
        x = s; y = lognorm.pdf(s,shape,loc=loc,scale=scale)
        l = f"{labels[i]} lognormal fit (ML)"
        fPlot = ax[0].plot(
            x,y,label=l, # Maximum likelyhood
            color=colours[i],
            linewidth=1
        ) # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»
        dicData['plots']['fig1']['fitPlot'+idx] = {'t':'function','x':x,'y':y,'l':l}


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

        x = csSort; y = ccdfEmp;
        l = f"{labels[i]} empirical CCDF"
        ax[1].plot(
            x,y,
            marker="o",
            color=colours[i],
            linewidth=1,
            # linestyle="--",
            linestyle="none",
            label=l,
            alpha=0.5
        )
        dicData['plots']['fig2']['scatterPlot'+idx] = {'t':'scatter','x':x,'y':y,'l':l}

        x = csSort; y = ccdfFit;
        l = fr"{labels[i]} pareto fit$"
        ax[1].plot(
            x,y,label=l,
            color=colours[i]
        )
        dicData['plots']['fig2']['fitPlot'+idx] = {'t':'function','x':x,'y':y,'l':l}

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

    # Style

    xl = r"$s$"; yl = r"$P(s)$"
    libF.SetPlotStyle(
        xl,yl,ax=ax[0],
        yNotation="sci" # ,xNotation="sci"
    )
    dicData['style']['fig1']['scale'] = {'x':'lin','y':'lin'}
    dicData['style']['fig1']['labels'] = {'x':xl,'y':yl}

    xl = r"$s$"; yl = r"$P(s)$"
    libF.SetPlotStyle(
        xl,ax=ax[1],
        xScale="log",yScale="log"
    )
    dicData['style']['fig2']['scale'] = {'x':'log','y':'log'}
    dicData['style']['fig2']['labels'] = {'x':xl,'y':yl}

    libF.CentrePlot()
    libF.SaveFig('CitySizeDistribution','KS',dicData)

def CityAverageFig(ca,Nt,dt):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}

    labels = ["Ext.","Apx."]
    colours = ["blue","red"]
    timeInterval = np.arange(Nt+1)*dt

    for i,k in enumerate(ca):
        x = timeInterval; y = ca[k]
        l = rf"{labels[i]} city size average $\langle s\rangle$"
        plt.plot(
            x,y,
            label=l,
            color=colours[i],
            linewidth=1,
        )
        dicData['plots']['functionPlot'+str(i)] = {'t':'function','x':x,'y':y,'l':l}

    libF.CentrePlot()

    xl = r"$t$"; yl = r"$\langle s\rangle$"
    libF.SetPlotStyle(xl,yl)
    dicData['style']['scale'] = {'x':'lin','y':'lin'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

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