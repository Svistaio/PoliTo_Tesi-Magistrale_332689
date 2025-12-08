import numpy as np

import importlib
from tqdm import tqdm

import libFigures as libF
importlib.reload(libF)

import matplotlib.pyplot as plt
libF.SetTextStyle()


### Main class ###

class KineticSimulation: # NetworkState
    def __init__(self,dicPrm,dicReg):
        #region Initial data dump
        self.li2Name = dicReg['li2Name']
        self.typ = ["ext","apx"]
        self.Ni = dicPrm['iterations']

        self.l = dicPrm['attractivity']
        self.a = dicPrm['convincibility']
        self.s = dicPrm['deviation']
        #endregion

        #region City size vectors
        self.Nc = dicReg['Nc']

        self.P = dicPrm['totalPop']
        self.Nt = dicPrm['stepNumber']
        self.dt = dicPrm['timeStep']

        self.vrtState   = { # Uniform initial state for all vertices
            self.typ[0]:np.ones((self.Nc,1),dtype=float)*(self.P/self.Nc),
            self.typ[1]:np.ones((self.Nc,1),dtype=float)*(self.P/self.Nc),
        }
        self.avrState    = {
            self.typ[0]:np.zeros((self.Nt+1,1),dtype=float),
            self.typ[1]:np.zeros((self.Nt+1,1),dtype=float)
        }

        for t in self.avrState: # Initial average state
            self.avrState[t][0] = self.P/self.Nc
        #endregion 

        #region Exact [Weighted] Adjacency matrix
        if dicPrm['edgeWeights']:
            T  = dicReg['W']/np.max(dicReg['W'])
        else:
            T  = dicReg['A']
        
        # Approximated adjacency matrix
        Mn = np.sum(T)
        wO = np.sum(T,axis=1,keepdims=True)
        wI = np.sum(T,axis=0,keepdims=True)
        R  = wO@wI/Mn
        # In my case wO=wI but it's better to define them in the most general way

        self.M = {self.typ[0]:T, self.typ[1]:R}
        #endregion

    def MonteCarloAlgorithm(self):
        Nt = self.Nt

        for nt in tqdm(range(Nt),desc="Updating states"):
            self.UpdateState(nt)

    def UpdateState(self,nt):
        dt = self.dt

        oldState = self.vrtState
        newState = {k: v.copy() for k,v in self.vrtState.items()}
 
        P = np.random.permutation(self.Nc)
        halfNc = int(np.floor(self.Nc/2))
        pi = P[:halfNc]; pr = P[halfNc+1:]

        for i in range(halfNc):
            for t in self.typ:
                # It's assumed node pi(i) is the interacting node while node ps(i) is the receiving one

                p = float(np.clip(self.M[t][pi[i],pr[i]]*dt,0,1))
                theta = np.random.binomial(1,p)
                si = oldState[t][pi[i]]; sr = oldState[t][pr[i]]

                E = self.NonLinearEmigration(si,sr)
                ga = self.StochasticFluctuations(E)

                newState[t][pi[i]] = si*(1-theta)+theta*si*(1-E+ga) 
                newState[t][pr[i]] = sr*(1-theta)+theta*(sr+si*E)

        self.vrtState = newState
        for t in self.typ:
            self.avrState[t][nt+1] = np.mean(newState[t])

    def NonLinearEmigration(
            self,
            si, # Interacting city size
            sr, # Receiving city size
        ):
        l = self.l; a = self.a

        if si != 0:
            rs = sr/si               # Relative population ratio
            ef = l*(rs**a)/(1+rs**a) # Actual emigration rate
        else:
            ef = 0
        return ef

    def StochasticFluctuations(self,E):
        sigma = self.s

        alpha = ((1-E)**2)/(sigma**2)
        theta = (sigma**2)/(1-E)

        if alpha<=1:
            raise ValueError("α must be >1 to have a non-degenerate gamma distribution, and thus always admissible fluctuations")

        ga = np.random.gamma(alpha,theta) # Initial sampling

        return ga+E-1 # Final left translation

    def SizeDistrFitsFig(self):
        cs = self.vrtState; Nc = self.Nc; Ni = self.Ni

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        dicData = libF.CreateDicData(2)

        labels = {self.typ[0]:'Ext.',self.typ[1]:'Apx.'}
        colours = {self.typ[0]:'blue',self.typ[1]:'red'}

        for i,t in enumerate(self.typ): # i[ndex] and t[ype]

            ### Lognormal fit ###
            libF.CreateHistogramPlot(
                cs[t],30,dicData['fig1'],
                l=f"{labels[t]} histogram",
                clr=colours[t],alfa=0.5,
                idx=i+1,ax=ax[0]
            ) # Histogram plot

            libF.CreateLognormalFitPlot(
                cs[t],dicData['fig1'],
                lAvr=fr"{labels[t]} mean value $\langle k\rangle$",
                lFit=f"{labels[t]} lognormal fit (ML)",
                clrAvr=colours[t],clrFit=colours[t],
                idx=i+1,ax=ax[0]
            ) # Fit plot


            ### Power law fit ###
            b = libF.CreateParetoFitPlot(
                cs[t],dicData['fig2'],
                lSct=f"{labels[t]} empirical CCDF",
                lFit=fr"{labels[t]} pareto fit",
                clrSct=colours[t],clrFit=colours[t],
                idx=i+1,ax=ax[1]
            )

            fig.text(
                .5,.975-i*.05,
                fr"{labels[t]}: $\quad s_{{min}}={np.min(cs[t]):.2f}\qquad$"
                fr"$s_{{max}}={np.max(cs[t]):.2f}\qquad$"
                fr"$\langle s\rangle ={np.mean(cs[t]):.2f}\qquad$"
                fr"$s_{{\Sigma}}={np.sum(cs[t]):.2f}\qquad$"
                fr"$\alpha={b:.2f}$",
                ha="center",
                # fontsize=10,
                color="black"
            )

        self.lbl = labels
        self.clr = colours

        si = np.argsort(cs['ext'],axis=0)
        for i in range(-1,-6,-1):
            fig.text(
                .625,.4+i*.03,
                fr'${self.li2Name[int(si[i])]}='
                fr'{int(cs['ext'][int(si[i])]):.2e}$'
                ,
                ha="center",
                # fontsize=10,
                color="black"
            )

        fig.text(.1,.95,fr"$Nc={Nc}\qquadNi={Ni}$")


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

    def SizeAverageFig(self):
        ca = self.avrState; Nt = self.Nt; dt = self.dt

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

    def SizeVsDegreeFig(self):
        cs = self.vrtState
        di = {t:np.sum(self.M[t],axis=0) for t in self.typ}

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        dicData = libF.CreateDicData(2)

        for i,scale in enumerate(['lin','log']):
            for t in self.typ:
                libF.CreateScatterPlot(
                    di[t],cs[t],dicData[f'fig{i+1}'],
                    l=self.lbl[t],
                    clr=self.clr[t],
                    idx=i+1,ax=ax[i]
                )

            libF.SetFigStyle(
                r'$k$',r'$cs(k)$',
                yScale=scale,
                data=dicData[f'fig{i+1}']
            )

        libF.CentreFig()
        libF.SaveFig('SizeVsDegree','KS',dicData)

    # def SizeDistrEvolutionFig(self):

### Discarded code ###

#region Old implementation for fluctuations with a [forcefully] resampled Gaussian until the value picked ensures the post-interaction population is positive
    # def StochasticFluctuations(sigma,E):
    #     while True:
    #         mu = np.random.normal(0,sigma,size=1)
    #         if mu>E-1: #and mu<E:
    #             break
    #     # The conditions «mu>E-1» and «mu<E» are necessary to have the total emigration rage 1-E+μ between 0 and 1
    #     return mu
#endregion I save it since I find it an interesting wrong approach
