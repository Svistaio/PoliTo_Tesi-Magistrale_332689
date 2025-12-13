
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

from tqdm import tqdm

import importlib
import libFigures as libF
importlib.reload(libF)


### Main class ###

class KineticSimulation(): # NetworkState
    def __init__(self,clsPrm,clsReg):
        #region Initial data dump
        self.li2Name = clsReg.li2Name
        self.Ni = clsPrm.iterations

        self.l = clsPrm.attractivity
        self.a = clsPrm.convincibility
        self.s = clsPrm.deviation

        self.Nc = clsReg.Nc
        self.P  = clsPrm.totalPop
        self.p0 = self.P/self.Nc

        self.Nt = clsPrm.stepNumber
        self.dt = clsPrm.timeStep
        self.Ns = 100 # Number of screenshots [not considering the initial state]
        #endregion

        #region City size vectors
        # Arbitrarly the index «0» is chosen to mean the exact simulation while «1» the approximate one
        self.typ = (0,1)
        self.lbl = ("Ext.","Apx.")
        self.clr = ("blue","red")

        # Uniform initial state for all vertices
        self.vrtState = np.ones((self.Nc,2),dtype=float)*self.p0
        self.avrState = np.ones((self.Ns+1,2),dtype=float)*self.p0
        self.screenshots = np.ones((self.Nc,self.Ns+1,2),dtype=float)*self.p0

        self.ns = np.array([i*self.Nt/self.Ns for i in range(self.Ns+1)])
        self.times = self.ns*self.dt
        #endregion 

        #region Exact [Weighted] Adjacency matrix
        self.di = np.sum(clsReg.A,axis=0)
        self.M = np.zeros((self.Nc,self.Nc,2),dtype=float)

        if clsPrm.edgeWeights:
            self.M[:,:,0] = clsReg.W/np.max(clsReg.W)
        else:
            self.M[:,:,0] = clsReg.A
        
        # Approximated adjacency matrix
        Mn = np.sum(self.M[:,:,0])
        wO = np.sum(self.M[:,:,0],axis=1,keepdims=True)
        wI = np.sum(self.M[:,:,0],axis=0,keepdims=True)
        self.M[:,:,1]  = wO@wI/Mn
        # In this case wO=wI but it's better to define them in the most general way
        #endregion

        self.figData = libF.FigData(clsPrm,'KS')

    # Simulation
    def MonteCarloAlgorithm(self):
        ns2i = {ns:i for i,ns in enumerate(self.ns)}
        # dt = self.dt; M = self.M

        def UpdateState(vrtState): 
           return EvolveState(
                vrtState,self.Nc,self.dt,
                self.M,self.l,self.a,self.di,
                self.s,np.array(self.typ)
            )

        for nt in tqdm(range(self.Nt),desc="Updating states"):
            self.vrtState = UpdateState(self.vrtState)
            if nt+1 in ns2i:
                for t in self.typ:
                    self.avrState[ns2i[nt+1],t] = np.mean(self.vrtState[:,t])
                    self.screenshots[:,ns2i[nt+1],t] = self.vrtState[:,t].copy()

        self.WriteSimulationData()

    def WriteSimulationData(self):
        import zipfile as zf
        from zipfile import ZipFile as ZF
        from io import StringIO as sio
        import json

        typ = self.typ; li2Name = self.li2Name
        lbl = self.lbl; vrtState = self.vrtState

        si = np.argsort(vrtState,axis=0)
        dicName2SortedPop = {
            t:{
                li2Name[i]:vrtState[i,t] for i in si[::-1,t]
            } for t in typ
        }
        self.si = si

        with ZF(
            '../Dati/SimulationData.zip','w',
            compression=zf.ZIP_DEFLATED,compresslevel=9
        ) as z:
            for t in typ:
                buf = sio()
                name = f'{lbl[t]}CitySizesFinal.json'
                json.dump(list(vrtState[:,t]),buf)
                value = buf.getvalue()
                z.writestr(name,value)

                buf = sio()
                name = f'{lbl[t]}CitySizesSorted.json'
                json.dump(dicName2SortedPop[t],buf)
                value = buf.getvalue()
                z.writestr(name,value)

    # Figures
    def SizeDistrFittingsFig(self):
        cs = self.vrtState; typ = self.typ
        lbl = self.lbl; clr = self.clr
        Nc = self.Nc; Ni = self.Ni
        figData = self.figData

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        figData.SetFigs(2)

        for t in typ: # t[ype]
            sMax = np.max(cs[:,t]); sMin = np.min(cs[:,t])
            bins = np.logspace(np.log10(sMin),np.log10(sMax),25)

            ### Lognormal fit ###
            libF.CreateHistogramPlot(
                cs[:,t],bins,figData.fig1,
                l=f"{lbl[t]} histogram",
                clr=clr[t],alfa=0.5,
                idx=t+1,ax=ax[0]
            ) # Histogram plot
            libF.CreateLognormalFitPlot(
                cs[:,t],figData.fig1,
                lAvr=fr"{lbl[t]} mean value $\langle k\rangle$",
                lFit=f"{lbl[t]} lognormal fit (ML)",
                clrAvr=clr[t],clrFit=clr[t],
                idx=t+1,ax=ax[0]
            ) # Fit plot

            ### Power law fit ###
            b = libF.CreateParetoFitPlot(
                cs[:,t],figData.fig2,
                lSct=f"{lbl[t]} empirical CCDF",
                lFit=fr"{lbl[t]} pareto fit",
                clrSct=clr[t],clrFit=clr[t],
                idx=t+1,ax=ax[1]
            )

            fig.text(
                .5,.975-t*.05,
                fr"{lbl[t]}: $\quad s_{{min}}={np.min(cs[:,t]):.2f}\qquad$"
                fr"$s_{{max}}={np.max(cs[t]):.2f}\qquad$"
                fr"$\langle s\rangle ={np.mean(cs[t]):.2f}\qquad$"
                fr"$s_{{\Sigma}}={np.sum(cs[t]):.2f}\qquad$"
                fr"$\alpha={b:.2f}$",
                ha="center",color="black"#,fontsize=10,
            )

        fig.text(.1,.95,fr"$Nc={Nc}\qquadNi={Ni}$")


        # Style
        libF.SetFigStyle(
            r"$cs$",r"$P(cs)$",
            yNotation="sci",xScale='log', # ,xNotation="sci"
            ax=ax[0],data=figData.fig1
        )

        libF.SetFigStyle(
            r"$cs$",r"$P(cs)$",
            xScale="log",yScale="log",
            ax=ax[1],data=figData.fig2
        )


        libF.CentreFig()
        figData.SaveFig('SizeDistributionFittings')

    def AverageSizeFig(self):
        ca = self.avrState
        lbl = self.lbl; clr = self.clr
        times = self.times
        figData = self.figData

        fig = plt.figure()
        figData.SetFigs(1)

        # np.convolve()
        for t in self.typ:
            libF.CreateFunctionPlot(
                times,ca[:,t],
                figData.fig,
                l=rf"{lbl[t]} city size average $\langle s\rangle$",
                clr=clr[t],idx=t+1
            )

        libF.CentreFig()
        libF.SetFigStyle(
            r"$t$",r"$\langle cs\rangle$",
            data=figData.fig
        )
        figData.SaveFig('AverageSize')

    def SizeVsDegreeFig(self):
        cs = self.vrtState; typ = self.typ
        lbl = self.lbl; clr = self.clr
        di = self.di; si = self.si
        li2Name = self.li2Name
        figData = self.figData

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        figData.SetFigs(2)

        for i,scale in enumerate(['lin','log']):
            for t in typ:
                libF.CreateScatterPlot(
                    di,cs[:,t],
                    getattr(figData,f'fig{i+1}'),
                    l=lbl[t],clr=clr[t],
                    idx=i+1,ax=ax[i]
                )

            libF.SetFigStyle(
                r'$k$',r'$cs(k)$',
                yScale='log',xScale=scale,ax=ax[i],
                data=getattr(figData,f'fig{i+1}')
            )

        for i in range(-1,-6,-1):
            fig.text(
                .4,.3+i*.03,
                fr'${li2Name[si[i,0]]}='
                fr'{cs[si[i,0],0]:.2e}$'
                ,
                ha="center",
                # fontsize=10,
                color="black"
            )

        libF.CentreFig()
        figData.SaveFig('SizeVsDegree')

    def SizeDistrEvolutionFig(self):
        screenshots = self.screenshots
        typ = self.typ; dt = self.dt
        ns = self.ns; Ns = self.Ns
        figData = self.figData

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')
        figData.SetFigs(2)

        samples = 6
        clrmap = plt.get_cmap('plasma')
        colours = [clrmap(i/(samples-1)) for i in range(samples)]

        sMax = np.max(screenshots[:,-1,:])
        # bins = np.linspace(0,sMax,26)
        sMin = np.min(screenshots[:,-1,:])
        bins = np.logspace(np.log10(sMin),np.log10(sMax),26)

        for t in typ: # t[ype]
            for j,s in enumerate(np.linspace(0,Ns,samples,dtype=int)):
                libF.CreateHistogramPlot(
                    screenshots[:,s,t],bins,
                    getattr(figData,f'fig{t+1}'),
                    l=f"t = {int(ns[s]*dt)}",
                    clr=colours[j],alfa=0.4,
                    idx=j+1,ax=ax[t],norm=False
                ) # Histogram plot

            libF.SetFigStyle(
                r"$cs$",r"$P(cs)$",
                yNotation="sci", # ,xNotation="sci"
                xScale='log',ax=ax[t],
                data=getattr(figData,f'fig{t+1}')
            ) # Style

        libF.CentreFig()
        figData.SaveFig('SizeDistributionEvolution')

    def SizeEvolutionsFig(self):
        screenshots = self.screenshots
        Nc = self.Nc; typ = self.typ
        figData = self.figData
        times = self.times

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')
        figData.SetFigs(2)

        clrmap = plt.get_cmap('plasma')
        colours = [clrmap(i/(Nc-1)) for i in range(Nc)]

        # sMax = np.max(screenshots[:,-1,:])

        for t in typ: # t[ype]
            for c in range(Nc):
                libF.CreateFunctionPlot(
                    times,screenshots[c,:,t],#/sMax
                    getattr(figData,f'fig{t+1}'),
                    clr=colours[c],#alfa=0.4,
                    idx=c+1,ax=ax[t]
                ) # Histogram plot

            libF.SetFigStyle(
                r"$t$",r"$s$",
                # yNotation="sci", # ,xNotation="sci"
                yScale='log',ax=ax[t],
                data=getattr(figData,f'fig{t+1}')
            ) # Style

        libF.CentreFig()
        figData.SaveFig('SizeEvolutions')


### Auxiliary functions ###

@njit
def EvolveState(oldState,Nc,dt,M,l,a,di,s,typ):
    newState = oldState.copy()

    P = np.random.permutation(Nc)
    halfNc = int(np.floor(Nc/2))
    pi = P[:halfNc]; pr = P[halfNc:]

    for i in range(halfNc):
        for t in typ:
            # It's assumed node pi(i) is the interacting node while node ps(i) is the receiving one

            p = M[pi[i],pr[i],t]*dt
            p = 1 if p>1 else (0 if p<0 else p)

            theta = np.random.binomial(1,p)
            si = oldState[pi[i],t]; sr = oldState[pr[i],t]

            E = NonLinearEmigration(si,pi[i],sr,pr[i],l,a,di)
            ga = StochasticFluctuations(s,E)

            newState[pi[i],t] = si*(1-theta)+theta*si*(1-E+ga) 
            newState[pr[i],t] = sr*(1-theta)+theta*(sr+si*E)

    return newState

@njit
def NonLinearEmigration(
        si,ii, # Interacting city size
        sr,ir, # Receiving city size
        l,a,di
    ):

    if si != 0:
        # 4
        # rsk = (sr/si)*(di[ir]/di[ii]) # Relative population ratio
        # efl = l*(rsk**a)/(1+rsk**a)    # Actual emigration rate for the lumping fraction

        # rsk = (si/sr)*(di[ii]/di[ir]) # Inverse relative population ratio
        # efs = l*(rsk**a)/(1+rsk**a)    # Actual emigration rate for the separation fraction

        # z = .1
        # return (1-z)*efl+z*efs

        #3
        rsk = (sr/si)*(di[ir]/di[ii]) # Relative population ratio
        return l*(rsk**a)/(1+rsk**a)   # Actual emigration rate

        # 2
        # rs = (sr/si)*(di[ir]/di[ii]) # Relative population ratio
        # return l*(rsk/a)/(1+rsk/a)     # Actual emigration rate

        # 1
        # rs = sr/si                 # Relative population ratio
        # return l*(rs**a)/(1+rs**a) # Actual emigration rate
    else:
        return 0

@njit
def StochasticFluctuations(sigma,E):
    alpha = ((1-E)**2)/(sigma**2)
    theta = (sigma**2)/(1-E)

    if alpha<=1:
        raise ValueError("α must be >1 to have a non-degenerate gamma distribution, and thus always admissible fluctuations")

    ga = np.random.gamma(alpha,theta) # Initial sampling

    return ga+E-1 # Final left translation

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
