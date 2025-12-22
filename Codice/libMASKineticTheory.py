
import numpy as np
from scipy import stats
from numba import njit

import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
# from tqdm import tqdm

from matplotlib.pyplot import get_cmap
from matplotlib.colors import LogNorm

from libGUI import ProgressGUI
from libExtractData import WriteSimulationData

import importlib
import libFigures as libF
importlib.reload(libF)


### Main class ###

class KineticSimulation():
    def __init__(self,clsPrm,clsReg):
        self.il = int(clsPrm.interactingLaw)

        self.l = float(clsPrm.attractivity)
        self.a = float(clsPrm.convincibility)
        self.s = float(clsPrm.deviation)
        self.z = float(clsPrm.zetaFraction)

        self.Ni = int(clsPrm.iterations)
        Nc = int(clsReg.Nc); self.Nc = Nc
        self.li2Name = clsReg.li2Name

        self.progressBar = clsPrm.progressBar
        dt = float(clsPrm.timestep); self.dt = dt
        Nt = int(clsPrm.timesteps); self.Nt = Nt

        Ns = int(clsPrm.screenshots); self.Ns = Ns
        ns = np.array(
            [i*Nt/Ns for i in range(Ns+1)],
            dtype=np.int64
        ); self.ns = ns

        sf = clsPrm.smoothingFactor; self.sf = sf
        ks = int(Ns/sf); self.ks = ks # Kernerl size
        self.times = ns[::ks]*dt

        di = np.sum(
            clsReg.A,axis=1,
            dtype=np.int64,
            keepdims=True
        ); self.di = di.reshape((Nc,))
        # Inverse degrees
        invdi = np.array(1/di,dtype=np.float64)
        self.D = di@invdi.T

        self.R = int(clsPrm.region)
        self.P = int(clsPrm.population)
        self.p0 = float(self.P/self.Nc)

        M = np.zeros((Nc,Nc,2),dtype=np.float64)

        # Exact [Weighted] Adjacency matrix
        if clsPrm.edgeWeights:
            M[:,:,0] = clsReg.W/np.max(clsReg.W)
        else:
            M[:,:,0] = clsReg.A
        
        # Approximated adjacency matrix
        Mn = np.sum(M[:,:,0])
        wO = np.sum(M[:,:,0],axis=1,keepdims=True)
        wI = np.sum(M[:,:,0],axis=0,keepdims=True)
        M[:,:,1]  = wO@wI/Mn
        # In this case wO=wI but it's better to define them in the most general way

        self.Mdt = M*dt
        self.typ = np.array([0,1],dtype=np.int64)

        self.lbl = ('Ext.','Apx.')
        self.clr = ('#0072B2','#D55E00')
        # self.clr = ('blue','red')

        self.figData = libF.FigData(clsPrm,'KS')

    # Simulation
    def MonteCarloSimulation(self):
        Ni = self.Ni
        Nc = self.Nc
        p0 = self.p0

        progressBar = self.progressBar
        Nt = self.Nt
        Ns = self.Ns
        ns = self.ns

        D = self.D

        Mdt = self.Mdt
        l = self.l
        a = self.a
        s = self.s
        z = self.z
        il = self.il
        typ = self.typ.size

        process = range(Ni)

        if progressBar: 
            progress = np.zeros(Ni,dtype=np.int64)
            elapsed = np.zeros(Ni,dtype=np.float64)
            done = np.zeros(Ni,dtype=np.int8)

            shmp = shared_memory.SharedMemory(create=True,size=progress.nbytes)
            shme = shared_memory.SharedMemory(create=True,size=elapsed.nbytes)
            shmd = shared_memory.SharedMemory(create=True,size=done.nbytes)

            np.ndarray(progress.shape,progress.dtype,shmp.buf)[:] = 0
            np.ndarray(elapsed.shape,elapsed.dtype,shme.buf)[:] = 0
            np.ndarray(done.shape,done.dtype,shmd.buf)[:] = 0

            bar = ProgressGUI(
                Ni,Nt,
                shmp.name,
                shme.name,
                shmd.name
            )

            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=3,
                mp_context=ctx
            ) as executor:
                futures = {}
                for p in process:
                    futures[p] = executor.submit(
                        MonteCarloAlgorithm,
                        p,Ni,Nc,Ns,ns,
                        p0,D,Mdt,typ,
                        l,a,s,z,il,
                        gui=True,
                        namep=shmp.name,
                        namee=shme.name,
                        named=shmd.name
                    )
                bar.mainloop()


            shmp.close(); shmp.unlink()
            shme.close(); shme.unlink()
            shmd.close(); shmd.unlink()

        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=3,
                mp_context=ctx
            ) as executor:
                futures = {}
                for p in process:
                    futures[p] = executor.submit(
                        MonteCarloAlgorithm,
                        p,Ni,Nc,Ns,ns,
                        p0,D,Mdt,typ,
                        l,a,s,z,il,
                    )

        data = [None]*Ni
        for p, fut in futures.items():
            data[p] = fut.result()
        self.data = data

        self.EvaluateSimulationData()
        WriteSimulationData(
            self.lbl,
            self.siVrt,
            self.vrtState,
            self.typ,
            self.li2Name,
            Ni
        )

    def EvaluateSimulationData(self):
        Ni = self.Ni
        data = self.data
        sf = self.sf
        ks = self.ks

        def Convolve(v):
            Nrw, _, Nty = v.shape # Number of rows, times and types

            filter = np.array([float(1/ks)]*ks)
            sv = np.zeros((Nrw,sf+1,Nty),dtype=float) # Smoothed vector

            for r in range(Nrw):
                for t in range(Nty):
                    sv[r,0,t] = v[r,0,t].copy()
                    conv = np.convolve(
                        v[r,1:,t],
                        filter,
                        'valid'
                    )
                    sv[r,1:,t] = conv[::ks].copy()
            
            # This functions applies the mean every ks unit of the Ns screenshots
            return sv

        self.vrtState = np.array([data[p][0] for p in range(Ni)])
        screenshots = np.array([data[p][1] for p in range(Ni)])
        self.avrState = Convolve(np.mean(screenshots,axis=1))

        self.nodeAvrVrtState = np.mean(self.vrtState,axis=1)
        self.nodeMinVrtState = np.min(self.vrtState,axis=1)
        self.nodeMaxVrtState = np.max(self.vrtState,axis=1)
        self.nodeSumVrtState = np.sum(self.vrtState,axis=1)

        if Ni>1:
            self.ta = stats.t.ppf(0.975,df=Ni-1)

            self.itAvrVrtState = np.mean(self.vrtState,axis=0)
            self.itAvrScreenshots = np.mean(screenshots,axis=0)
        else:
            self.ta = None

            self.itAvrVrtState = self.vrtState[0,:,:]
            self.itAvrScreenshots = screenshots[0,:,:,:]

        self.itAvrConvScreenshots = Convolve(self.itAvrScreenshots)
        self.siVrt = np.argsort(self.vrtState,axis=1)
        self.siAvr = np.argsort(self.itAvrVrtState,axis=0)

    # Figures
    def SizeDistrFittingsFig(self):
        Ni = self.Ni
        ta = self.ta

        cs = self.vrtState
        csAvr = self.nodeAvrVrtState
        csMin = self.nodeMinVrtState
        csMax = self.nodeMaxVrtState
        csSum = self.nodeSumVrtState

        typ = self.typ
        lbl = self.lbl
        clr = self.clr

        figData = self.figData
        fig, ax = figData.SetFigs(1,3,size=(23,6))

        sMax = cs.max(); sMin = cs.min()

        for t in typ: # t[ype]
            ### Lognormal fit ###
            libF.CreateHistogramPlot(
                cs[:,:,t],30,
                figData.fig1,
                limits=(sMin,sMax),
                xScale='log',
                Ni=Ni,
                ta=ta,
                label=f"{lbl[t]} histogram",
                color=clr[t],
                alpha=(0.35,0.45) if Ni>1 else 0.35,
                idx=t+1,
                ax=ax[0]
            ) # Histogram plot

            libF.CreateLognormalFitPlot(
                cs[:,:,t],
                figData.fig1,
                limits=(sMin,sMax),
                xScale='log',
                Ni=Ni,
                ta=ta,
                label=(
                    fr"{lbl[t]} mean value $\langle k\rangle$",
                    f"{lbl[t]} lognormal fit (ML)"
                ),
                color=(clr[t],clr[t]),
                alpha=(1,0.15) if Ni>1 else 1,
                idx=t+1,
                ax=ax[0]
            ) # Fit plot

            ### Power law fit log-log ###
            b = libF.CreateParetoFitPlot(
                cs[:,:,t],
                figData.fig2,
                upperbound=sMax,
                yScale='log',
                Ni=Ni,
                ta=ta,
                label=(
                    f"{lbl[t]} empirical CCDF",
                    fr"{lbl[t]} pareto fit"
                ),
                color=(clr[t],clr[t]),
                alpha=((0.6,0.3),(1,0.15)) if Ni>1 else (0.6,1),
                idx=t+1,
                ax=ax[1]
            )

            ### Power law fit log-lin ###
            b = libF.CreateParetoFitPlot(
                cs[:,:,t],
                figData.fig3,
                upperbound=sMax,
                yScale='lin',
                Ni=Ni,
                ta=ta,
                label=(
                    f"{lbl[t]} empirical CCDF",
                    fr"{lbl[t]} pareto fit"
                ),
                color=(clr[t],clr[t]),
                alpha=((0.6,0.3),(1,0.15)) if Ni>1 else (0.6,1),
                idx=t+1,
                ax=ax[2]
            )

            fig.text(
                .5,.975-t*.05,
                fr'{lbl[t]}:$\quad$'+
                DataString(csMin[:,t],Ni,ta,r's_{{min}}')+
                DataString(csMax[:,t],Ni,ta,r's_{{max}}')+
                DataString(csAvr[:,t],Ni,ta,r'\langle s\rangle')+
                DataString(
                    csSum[:,t],Ni,ta,r's_{{\Sigma}}',
                    formatVal='.2e',formatErr='.2e'
                )+
                DataString(b,Ni,ta,r'\beta',space=False),
                ha='center',color=clr[t]
            )

        fig.text(.1,.975,fr'$Nc={self.Nc}$',ha='center')
        fig.text(.1,.925,fr'$R={self.R}$',ha='center')

        p = .95; i = 0; d = 0.035
        fig.text(p-i*d,.975,fr'$Ns={self.Ns}$',ha='center')
        fig.text(p-i*d,.925,fr'$sf={self.sf}$',ha='center')

        i += 1
        fig.text(p-i*d,.975,fr'$Ni={Ni}$',ha='center')
        fig.text(p-i*d,.925,fr'$dt={self.dt}$',ha='center')

        i += 1
        fig.text(p-i*d,.975,fr'$\sigma={self.s}$',ha='center')
        fig.text(p-i*d,.925,fr'$il={self.il}$',ha='center')

        i += 1
        fig.text(p-i*d,.975,fr'$\lambda={self.l}$',ha='center')
        fig.text(p-i*d,.925,fr'$\alpha={self.a}$',ha='center')

        i += 1
        if self.il in (2,4,6):
            fig.text(p-i*d,.95,fr'$ζ={self.z}$',ha='center')

        # Style
        libF.SetFigStyle(
            r'$cs$',r'$P(cs)$',
            yNotation='sci',xScale='log', # ,xNotation="sci"
            ax=ax[0],data=figData.fig1
        )

        libF.SetFigStyle(
            r'$cs$',r'$P(cs)$',
            xScale='log',yScale='log',
            ax=ax[1],data=figData.fig2
        )

        libF.SetFigStyle(
            r'$cs$',r'$P(cs)$',
            xScale='log',yScale='lin',
            ax=ax[2],data=figData.fig3
        )

        # CentreFig()
        figData.SaveFig('SizeDistributionFittings')

    def AverageSizeFig(self):
        Ni = self.Ni
        ta = self.ta

        times = self.times
        ca = self.avrState
        
        lbl = self.lbl
        clr = self.clr

        figData = self.figData
        fig = figData.SetFigs()

        for t in self.typ:
            libF.CreateFunctionPlot(
                times,
                ca[:,:,t],
                figData.fig,
                Ni=Ni,
                ta=ta,
                label=rf"{lbl[t]} city size average $\langle s\rangle$",
                color=clr[t],
                alpha=(1,0.15) if Ni>1 else 1,
                idx=t+1
            )

        # CentreFig()
        libF.SetFigStyle(
            r"$t$",r"$\langle cs\rangle$",
            data=figData.fig
        )
        figData.SaveFig('AverageSize')

    def SizeVsDegreeFig(self):
        di = self.di
        cs = self.itAvrVrtState

        typ = self.typ
        lbl = self.lbl
        clr = self.clr

        si = self.siAvr
        li2Name = self.li2Name

        figData = self.figData
        fig, ax = figData.SetFigs(1,2,size=(15,6))

        for i,scale in enumerate(['lin','log']):
            for t in typ:
                libF.CreateScatterPlot(
                    di,cs[:,t],
                    getattr(figData,f'fig{i+1}'),
                    label=lbl[t],
                    color=clr[t],
                    idx=t+1,
                    ax=ax[i]
                )

            libF.SetFigStyle(
                r'$k$',r'$cs(k)$',
                yScale='log',xScale=scale,
                data=getattr(figData,f'fig{i+1}'),
                ax=ax[i]
            )

        for t in typ:
            for i in range(-1,-6,-1):
                fig.text(
                    .4,.5-.2*t+i*.03,
                    DataString(
                        cs[si[i,t],t],
                        head=li2Name[si[i,t]],
                        formatVal='.2e',
                        space=False
                    ),
                    ha="center",
                    # fontsize=10,
                    color=clr[t]
                )

        # CentreFig()
        figData.SaveFig('SizeVsDegree')

    def SizeDistrEvolutionFig(self):
        screenshots = self.itAvrScreenshots

        ns = self.ns
        Ns = self.Ns

        dt = self.dt
        typ = self.typ

        figData = self.figData
        fig, ax = figData.SetFigs(1,2,size=(15,6))
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')

        samples = 6
        clrmap = get_cmap('inferno') # magma
        colours = [clrmap(i/(samples-1)) for i in range(samples)]

        sMax = np.max(screenshots[:,-1,:])
        sMin = np.min(screenshots[:,-1,:])

        for t in typ: # t[ype]
            for j,s in enumerate(np.linspace(0,Ns,samples,dtype=int)):
                libF.CreateHistogramPlot(
                    screenshots[:,s,t],21,
                    getattr(figData,f'fig{t+1}'),
                    limits=(sMin,sMax),
                    xScale='log',
                    label=f"t = {int(ns[s]*dt)}",
                    color=colours[j][:-1],
                    alpha=0.4,
                    idx=j+1,
                    ax=ax[t],
                    norm=False
                ) # Histogram plot

            libF.SetFigStyle(
                r"$cs$",r"$P(cs)$",
                yNotation="sci", # ,xNotation="sci"
                xScale='log',
                yScale='log',
                ax=ax[t],
                data=getattr(figData,f'fig{t+1}')
            ) # Style

        # CentreFig()
        figData.SaveFig('SizeDistributionEvolution')

    def SizeEvolutionsFig(self):
        # Nc = self.Nc
        typ = self.typ

        times = self.times
        screenshots = self.itAvrConvScreenshots

        di = self.di
        sf = self.sf

        figData = self.figData
        fig, ax = figData.SetFigs(1,2,size=(15,6))
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')

        dk, _ = np.unique(di,return_counts=True); Nk = dk.size
        screenshotsk = np.zeros((Nk,sf+1,2),dtype=np.float64)
        for i,k in enumerate(dk):
            bk = di == k
            screenshotsk[i,:,:] = screenshots[bk,:,:].mean(axis=0)

        clrmap = get_cmap('inferno') # magma
        norm = LogNorm(vmin=dk.min(),vmax=dk.max())
        colours = clrmap(norm(dk[::-1]))
        # colours = [clrmap(i/(Nc-1)) for i in range(Nc)]

        labels = ['']*Nk
        idx = np.linspace(0,Nk-1,6,dtype=np.int64)
        for i in idx: labels[i] = f'k={dk[i]}'

        for t in typ: # t[ype]
            # norm = LogNorm(
            #     vmin=screenshotsk[:,-1,t].min(),
            #     vmax=screenshotsk[:,-1,t].max()
            # )
            # colours = clrmap(norm(screenshotsk[:,-1,t]))

            for k in range(Nk):
                libF.CreateFunctionPlot(
                    times,screenshotsk[k,:,t],
                    getattr(figData,f'fig{t+1}'),
                    color=colours[k,:],
                    alpha=0.8,
                    linewidth=1,
                    label=labels[k],
                    idx=k+1,
                    ax=ax[t]
                ) # Histogram plot

            libF.SetFigStyle(
                r"$t$",r"$s$",
                # yNotation="sci", # ,xNotation="sci"
                yScale='log',ax=ax[t],
                data=getattr(figData,f'fig{t+1}')
            ) # Style

        # CentreFig()
        figData.SaveFig('SizeEvolutions')

    def ShowFig(self):
        from matplotlib.pyplot import show
        show()


### Auxiliary functions ###

def MonteCarloAlgorithm(
    p,Ni,Nc,Ns,ns,
    p0,D,Mdt,typ,
    l,a,s,z,il,
    gui=False,
    namep=None,
    namee=None,
    named=None
):
    # rng = np.random.default_rng() # Even though it's recommended by Numpy it is not efficiently implemented in Numba, hence it halves the iterations per second if used

    # Uniform initial state for all vertices
    vrtState = np.full((Nc,typ),p0,dtype=np.float64)
    screenshots = np.full((Nc,Ns+1,typ),p0,dtype=np.float64)

    P = np.arange(Nc,dtype=np.int64)

    nk = ns[1]
    nsid = 1

    if gui: # If ProgressGUI is required
        # if namep is None or namee is None or named is None:
        #     raise ValueError('GUI requires namep, namee and named for shared memory')

        shmp = shared_memory.SharedMemory(name=namep)
        shme = shared_memory.SharedMemory(name=namee)
        shmd = shared_memory.SharedMemory(name=named)

        try:
            progress = np.ndarray((Ni,),dtype=np.int64,buffer=shmp.buf)
            elapsed = np.ndarray((Ni,),dtype=np.float64,buffer=shme.buf)
            done = np.ndarray((Ni,),dtype=np.int8,buffer=shmd.buf)

            EvolveState(
                vrtState,P,Nc,
                nk,Mdt,D,il,
                l,a,s,z#,rng
            ) # Warm-up iteration to avoid polluting the initial time t0
            screenshots[:,nsid,0] = vrtState[:,0]
            screenshots[:,nsid,1] = vrtState[:,1]
            nsid += 1

            t0 = time.perf_counter()
            for ns in range(Ns-1):
                EvolveState(
                    vrtState,P,Nc,
                    nk,Mdt,D,il,
                    l,a,s,z#,rng
                )

                progress[p] = (ns+2)*nk
                elapsed[p] = time.perf_counter()-t0
                # q.put((p,(ns+2)*nk,time.perf_counter()-t0))

                screenshots[:,nsid,0] = vrtState[:,0]
                screenshots[:,nsid,1] = vrtState[:,1]
                nsid += 1

            progress[p] = Ns*nk
            elapsed[p] = time.perf_counter()-t0
            done[p] = True
            # q.put((p,'done',time.perf_counter()-t0))

            return vrtState, screenshots

        finally:
            shmp.close()
            shme.close()
            shmd.close()

    else:
        EvolveState(
            vrtState,P,Nc,
            nk,Mdt,D,il,
            l,a,s,z#,rng
        ) # Warm-up iteration to avoid polluting the initial time t0
        screenshots[:,nsid,0] = vrtState[:,0]
        screenshots[:,nsid,1] = vrtState[:,1]
        nsid += 1

        t0 = time.perf_counter()
        for ns in range(Ns-1):
            EvolveState(
                vrtState,P,Nc,
                nk,Mdt,D,il,
                l,a,s,z#,rng
            )
            screenshots[:,nsid,0] = vrtState[:,0]
            screenshots[:,nsid,1] = vrtState[:,1]
            nsid += 1

        return vrtState, screenshots

@njit(cache=True)
def EvolveState(
    cs,P,Nc,nk,
    Mdt,D,il,
    l,a,s,z#,rng
):
    for _ in range(nk):
        FYDInPlaceShuffle(P,Nc)
        # P = np.random.permutation(Nc)
        # pi = P[:hNc]; pr = P[hNc:]

        hNc = Nc//2 # Nc half
        for i in range(hNc):
            ii = P[i]; ir = P[i+hNc]

            # t = 0
            p = Mdt[ii,ir,0]
            theta = np.random.random() < p

            if theta == 1:
                si = cs[ii,0]; sr = cs[ir,0]

                e = NonLinearEmigration(
                    si,sr,D[ir,ii],
                    il,l,a,z
                )
                ga = StochasticFluctuations(s,e)

                cs[ii,0] = si*(1-e+ga) 
                cs[ir,0] = sr+si*e

            # t = 1
            p = Mdt[ii,ir,1]
            theta = np.random.random() < p

            if theta == 1:
                si = cs[ii,1]; sr = cs[ir,1]

                e = NonLinearEmigration(
                    si,sr,D[ir,ii],
                    il,l,a,z
                )
                ga = StochasticFluctuations(s,e)

                cs[ii,1] = si*(1-e+ga) 
                cs[ir,1] = sr+si*e

            # p = 1 if p>1 else (0 if p<0 else p)
            # theta = np.random.binomial(1,p)

@njit(cache=True)
def FYDInPlaceShuffle(v,n):
    for i in range(n-1,0,-1):
        j = np.random.randint(0,i+1)
        tmp = v[i]
        v[i] = v[j]
        v[j] = tmp

        # In randint i+1 is necessary to include the i-th index

    # Fisher–Yates–Durstenfeld [in-place] shuffle
    # https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm

@njit(cache=True)
def NonLinearEmigration(
        si,  # Interacting city size
        sr,  # Receiving city size
        Dri, # Matrix of ratios dr/di
        il,l,a,z
    ):
    if si <= 0: return 0

    if il == 0:
        rs = sr/si
        return l*(rs**a)/(1+rs**a)

    elif il == 1:
        rsk = (sr/si)*Dri
        return l*(rsk/a)/(1+rsk/a)

    elif il == 2:
        if sr <= 0: return 0

        rsk = (sr/si)*Dri
        efl = l*(rsk/a)/(1+rsk/a)

        irsk = 1/rsk
        efs = l*(irsk/a)/(1+irsk/a)

        return (1-z)*efl+z*efs

    elif il == 3:
        rsk = (sr/si)*Dri
        return l*(rsk**a)/(1+rsk**a)

    elif il == 4:
        if sr <= 0: return 0

        rsk = (sr/si)*Dri
        efl = l*(rsk**a)/(1+rsk**a)

        irsk = 1/rsk
        efs = l*(irsk**a)/(1+irsk**a)

        return (1-z)*efl+z*efs

    elif il == 5:
        rsk = (sr/si)*Dri
        return l*(rsk/(1+rsk))**a

    else:
        if sr <= 0: return 0

        rsk = (sr/si)*Dri
        efl = l*(rsk/(1+rsk))**a

        irsk = 1/rsk
        efs = l*(irsk/(1+irsk))**a

        return (1-z)*efl+z*efs

        # rsk = (sr/si)*(dr*idi)  # Relative population*degree ratio
        # irsk = (si/sr)*(di*idr) # Inverse relative population*degree ratio
        # efl                     # Actual emigration rate for the lumping fraction
        # efs                     # Actual emigration rate for the separation fraction

    # Numba does not support officially the match structure, hence, even if it appears to work, it's better to use an «if/elif/else» one the sake of performance and stability

@njit(cache=True)
def StochasticFluctuations(sigma,E):
    alpha = ((1-E)**2)/(sigma**2)
    theta = (sigma**2)/(1-E)

    # if alpha<=1:
    #     raise ValueError("α must be >1 to have a non-degenerate gamma distribution, and thus always admissible fluctuations")

    ga = np.random.gamma(alpha,theta) # Initial sampling

    return ga+E-1 # Final left translation

def DataString(
    data,
    Ni=1,
    ta=None,
    head='',
    formatVal='.2f',
    formatErr='.2f',
    space=True
):
    (value,error) = libF.EvaluateConfidenceInterval(data,ta,Ni)
    
    space = r'\qquad' if space else ''
    if error is None:
        return fr'${head}={value:{formatVal}}{space}$'
    else:
        return fr'${head}={value:{formatVal}}\pm{error:{formatErr}}{space}$'


### Discarded code ###

#region Old implementation for fluctuations with a [forcefully] resampled Gaussian until the value picked ensures the post-interaction population is positive
"""
    def StochasticFluctuations(sigma,E):
        while True:
            mu = np.random.normal(0,sigma,size=1)
            if mu>E-1: #and mu<E:
                break
        # The conditions «mu>E-1» and «mu<E» are necessary to have the total emigration rage 1-E+μ between 0 and 1
        return mu
"""
#endregion I save it since I find it an interesting wrong approach

#region Old implementation for the progress bar from «MonteCarloAlgorithm()» insinde the terminal, which can only be used with one process
"""
    bar = tqdm(
        range(Nt),
        desc=f"Updating states {p}",
        position=p,
        leave=False,
        dynamic_ncols=True,
        # ascii=True,
        mininterval=0.2,
        smoothing=0.5,
        miniters=100
    )
    if (nt+1) % 100 == 0:
        bar.update(100)
"""
#endregion

#region Alternative with the «.map()» method in «MonteCarloSimulation()» which cannot be used with «libGUI.ProgessGUI()»
"""
    self.data = list(
        executor.map(
            RunSingleProcess,
            [clsPrm]*Ni,
            [clsReg]*Ni
        )
    )
"""
#endregion

#region Progress bar selection
"""
if clsPrm.progressBar:
else:
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {}
        for p in process:
            futures[p] = executor.submit(
                MonteCarloAlgorithm,
                clsPrm,clsReg,p,di,
                np.array(typ,dtype=np.int64)
            )
"""
#endregion