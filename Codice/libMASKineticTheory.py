
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import importlib
import libFigures as libF
importlib.reload(libF)


### Main class ###

class KineticSimulation: # NetworkState
    def __init__(self,clsPrm,clsReg):
        #region Initial data dump
        self.li2Name = clsReg.li2Name
        self.typ = ["ext","apx"]
        self.Ni = clsPrm.iterations

        self.l = clsPrm.attractivity
        self.a = clsPrm.convincibility
        self.s = clsPrm.deviation
        #endregion

        #region City size vectors
        self.Nc = clsReg.Nc
        self.P  = clsPrm.totalPop
        self.p0 = self.P/self.Nc

        self.Nt = clsPrm.stepNumber
        self.dt = clsPrm.timeStep
        self.Ns = 5000 # Number of screenshots [not considering the initial state]

        self.vrtState    = { # Uniform initial state for all vertices
            t:np.zeros((self.Nc,),dtype=float) for t in self.typ
        }
        self.avrState    = {
            t:np.zeros((self.Nt+1,),dtype=float) for t in self.typ
        }

        for t in self.avrState: # Initial average state
            self.vrtState[t][:] = self.p0
            self.avrState[t][0] = self.p0
        #endregion 

        #region Exact [Weighted] Adjacency matrix
        self.di = np.sum(clsReg.A,axis=0)

        if clsPrm.edgeWeights:
            T  = clsReg.W/np.max(clsReg.W)
        else:
            T  = clsReg.A
        
        # Approximated adjacency matrix
        Mn = np.sum(T)
        wO = np.sum(T,axis=1,keepdims=True)
        wI = np.sum(T,axis=0,keepdims=True)
        R  = wO@wI/Mn
        # In my case wO=wI but it's better to define them in the most general way

        self.M = {self.typ[0]:T, self.typ[1]:R}
        #endregion

        self.figData = libF.FigData(clsPrm,'KS')

    def MonteCarloAlgorithm(self):
        Nt = self.Nt; Ns = self.Ns
        
        self.screenshots = {
            t:np.ones((self.Nc,Ns+1),dtype=float)*self.p0 for t in self.typ
        }
        self.ns = [i*Nt/Ns for i in range(Ns+1)]
        ns2i = {ns:i for i,ns in enumerate(self.ns)}

        for nt in tqdm(range(Nt),desc="Updating states"):
            self.UpdateState(nt)
            if nt+1 in ns2i:
                for t in self.typ:
                    self.screenshots[t][:,ns2i[nt+1]] = self.vrtState[t].copy()

        self.WriteSimulationData()

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

                E = self.NonLinearEmigration(si,sr,pi[i],pr[i])
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
            ii,
            ir
        ):
        l = self.l; a = self.a; di = self.di

        if si != 0:
            rs = (sr/si)*(di[ir]/di[ii]) # Relative population ratio
            ef = l*(rs/a)/(1+rs/a) # Actual emigration rate

            # rs = sr/si # Relative population ratio
            # ef = l*(rs**a)/(1+rs**a) # Actual emigration rate
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

    def WriteSimulationData(self):
        import zipfile as zf
        from zipfile import ZipFile as ZF
        from io import StringIO as sio
        import json

        self.si = {t:np.argsort(self.vrtState[t],axis=0) for t in self.typ}
        dicName2SortedPop = {
            t:{
                self.li2Name[i]:self.vrtState[t][i] for i in self.si[t][::-1]
            } for t in self.typ
        }

        with ZF(
            '../Dati/SimualationData.zip','w',
            compression=zf.ZIP_DEFLATED,compresslevel=9
        ) as z:
            for t in self.typ:
                buf = sio()
                name = f'{t}CitySizesFinal.json'
                json.dump(list(self.vrtState[t]),buf)
                value = buf.getvalue()
                z.writestr(name,value)

                buf = sio()
                name = f'{t}CitySizesSorted.json'
                json.dump(dicName2SortedPop[t],buf)
                value = buf.getvalue()
                z.writestr(name,value)

    def SizeDistrFittingsFig(self):
        cs = self.vrtState; Nc = self.Nc; Ni = self.Ni

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        self.figData.SetFigs(2)

        labels = {self.typ[0]:'Ext.',self.typ[1]:'Apx.'}
        colours = {self.typ[0]:'blue',self.typ[1]:'red'}

        for i,t in enumerate(self.typ): # i[ndex] and t[ype]
            ### Lognormal fit ###
            libF.CreateHistogramPlot(
                cs[t],30,self.figData.fig1,
                l=f"{labels[t]} histogram",
                clr=colours[t],alfa=0.5,
                idx=i+1,ax=ax[0]
            ) # Histogram plot
            libF.CreateLognormalFitPlot(
                cs[t],self.figData.fig1,
                lAvr=fr"{labels[t]} mean value $\langle k\rangle$",
                lFit=f"{labels[t]} lognormal fit (ML)",
                clrAvr=colours[t],clrFit=colours[t],
                idx=i+1,ax=ax[0]
            ) # Fit plot

            ### Power law fit ###
            b = libF.CreateParetoFitPlot(
                cs[t],self.figData.fig2,
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
                ha="center",color="black"#,fontsize=10,
            )

        self.lbl = labels
        self.clr = colours

        fig.text(.1,.95,fr"$Nc={Nc}\qquadNi={Ni}$")


        # Style
        libF.SetFigStyle(
            r"$cs$",r"$P(cs)$",
            yNotation="sci",xScale='log', # ,xNotation="sci"
            ax=ax[0],data=self.figData.fig1
        )

        libF.SetFigStyle(
            r"$cs$",r"$P(cs)$",
            xScale="log",yScale="log",
            ax=ax[1],data=self.figData.fig2
        )


        libF.CentreFig()
        self.figData.SaveFig('SizeDistributionFittings')

    def AverageSizeFig(self):
        ca = {
            t:[self.avrState[t][int(i)] for i in self.ns] for t in self.typ
        }
        times = np.array(self.ns)*self.dt

        fig = plt.figure()
        self.figData.SetFigs(1)

        labels = ["Ext.","Apx."]
        colours = ["blue","red"]

        # np.convolve()
        k = 50
        for i,type in enumerate(ca):
            libF.CreateFunctionPlot(
                times[::k]+times[-1],
                ca[type][::k]+ca[type][-1],
                self.figData.fig,
                l=rf"{labels[i]} city size average $\langle s\rangle$",
                clr=colours[i],idx=i+1
            )

        libF.CentreFig()
        libF.SetFigStyle(
            r"$t$",r"$\langle cs\rangle$",
            data=self.figData.fig
        )
        self.figData.SaveFig('AverageSize')

    def SizeVsDegreeFig(self):
        cs = self.vrtState
        di = {t:self.di for t in self.typ}

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        self.figData.SetFigs(2)

        for i,scale in enumerate(['lin','log']):
            for t in self.typ:
                libF.CreateScatterPlot(
                    di[t],cs[t],
                    getattr(self.figData,f'fig{i+1}'),
                    l=self.lbl[t],
                    clr=self.clr[t],
                    idx=i+1,ax=ax[i]
                )

            libF.SetFigStyle(
                r'$k$',r'$cs(k)$',
                yScale='log',xScale=scale,ax=ax[i],
                data=getattr(self.figData,f'fig{i+1}')
            )

        for i in range(-1,-6,-1):
            fig.text(
                .4,.3+i*.03,
                fr'${self.li2Name[self.si['ext'][i]]}='
                fr'{cs['ext'][self.si['ext'][i]]:.2e}$'
                ,
                ha="center",
                # fontsize=10,
                color="black"
            )

        libF.CentreFig()
        self.figData.SaveFig('SizeVsDegree')

    def SizeDistrEvolutionFig(self):
        screenshots = self.screenshots

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')
        self.figData.SetFigs(2)

        samples = 6
        clrmap = plt.get_cmap('plasma')
        colours = [clrmap(i/(samples-1)) for i in range(samples)]

        sMax = np.max([np.max(screenshots[t][:,-1]) for t in self.typ])
        # bins = np.linspace(0,sMax,26)
        sMin = np.min([np.min(screenshots[t][:,-1]) for t in self.typ])
        bins = np.logspace(np.log10(sMin),np.log10(sMax),26)

        ts = [i*self.Nt/self.Ns for i in range(self.Ns+1)]

        for i,t in enumerate(self.typ): # i[ndex] and t[ype]
            for j,s in enumerate(np.linspace(0,self.Ns,samples,dtype=int)):
                libF.CreateHistogramPlot(
                    screenshots[t][:,s],bins,
                    getattr(self.figData,f'fig{i+1}'),
                    l=f"t = {int(ts[s]*self.dt)}",
                    clr=colours[j],alfa=0.4,
                    idx=j+1,ax=ax[i],norm=False
                ) # Histogram plot

            libF.SetFigStyle(
                r"$cs$",r"$P(cs)$",
                yNotation="sci", # ,xNotation="sci"
                xScale='log',ax=ax[i],
                data=getattr(self.figData,f'fig{i+1}')
            ) # Style


        libF.CentreFig()
        self.figData.SaveFig('SizeDistributionEvolution')

    def SizeEvolutionsFig(self):
        screenshots = self.screenshots
        Nc = self.Nc
        times = np.array(self.ns)*self.dt

        fig, ax = plt.subplots(1,2,figsize=(15,6))
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')
        self.figData.SetFigs(2)

        clrmap = plt.get_cmap('plasma')
        colours = [clrmap(i/(Nc-1)) for i in range(Nc)]

        sMax = np.max([np.max(screenshots[t][:,-1]) for t in self.typ])

        k = 50
        for i,t in enumerate(self.typ): # i[ndex] and t[ype]
            for c in range(Nc):
                libF.CreateFunctionPlot(
                    times[::k]+times[-1],
                    screenshots[t][c,:][::k]+screenshots[t][c,:][-1],#/sMax
                    getattr(self.figData,f'fig{i+1}'),
                    clr=colours[c],#alfa=0.4,
                    idx=c+1,ax=ax[i]
                ) # Histogram plot

            libF.SetFigStyle(
                r"$t$",r"$s$",
                # yNotation="sci", # ,xNotation="sci"
                yScale='log',ax=ax[i],
                data=getattr(self.figData,f'fig{i+1}')
            ) # Style


        libF.CentreFig()
        self.figData.SaveFig('SizeEvolutions')


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
