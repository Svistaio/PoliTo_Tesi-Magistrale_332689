
# Library to apply the Kinetic Theory for Multi-Agent Systems

from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from multiprocessing import shared_memory
import time
# from tqdm import tqdm

import numpy as np
from scipy import stats
from numba import njit

from matplotlib.pyplot import get_cmap
from matplotlib.colors import LogNorm

from libGUIs import ProgressBarsGUI
from libData import WriteSimulationData
from libParameters import workersShM
import libFigures as libF


### Main classes ###

class KineticSimulation():
    def __init__(self,clsPrm,clsReg):
        self.il = int(clsPrm.interactingLaw)

        self.l = float(clsPrm.attractivity)
        self.a = float(clsPrm.convincibility)
        self.s = float(clsPrm.deviation)
        self.f = clsPrm.fluctuations
        self.z = float(clsPrm.zetaFraction)

        self.Ni = int(clsPrm.iterations)
        Nc = int(clsReg.Nc); self.Nc = Nc
        self.li2Name = clsReg.li2Name

        self.progressBar = clsPrm.progressBar
        dt = float(clsPrm.timestep); self.dt = dt
        Nt = int(clsPrm.timesteps); self.Nt = Nt

        Ns = int(clsPrm.snapshots); self.Ns = Ns
        ns = np.array(
            [i*Nt/Ns for i in range(Ns+1)],
            dtype=np.int64
        ); self.ns = ns

        sf = clsPrm.smoothingFactor; self.sf = sf
        ks = int(Ns/sf); self.ks = ks # Kernerl size
        self.times = ns[::ks]*dt

        self.di = np.sum(clsReg.A,axis=1,dtype=np.int32)
        self.idi = np.array(1.0/self.di,dtype=np.float64) # Inverse degrees
        # self.D = di@invdi.T

        self.R = int(clsPrm.region)
        self.P = int(clsPrm.population)
        self.p0 = float(self.P/self.Nc)

        # Exact [Weighted] Adjacency matrix
        if clsPrm.edgeWeights:
            M = np.array(clsReg.W/np.max(clsReg.W),dtype=np.float64)
        else:
            M = np.array(clsReg.A,dtype=np.float64)
        
        # Approximated adjacency matrix
        Mn = np.sum(M[:,:])
        self.wOdt = np.sum(M[:,:],axis=1)*dt/Mn
        self.wI = np.sum(M[:,:],axis=0)
        # M[:,:,1]  = wO@wI/Mn
        # In this case wO=wI but it's better to define them rigorously

        self.Mdt = M*dt
        self.typ = np.array([0,1],dtype=np.int64)

        self.lbl = ('Ext.','Apx.')
        self.clr = ('#0072B2','#D55E00')
        # self.clr = ('blue','red')

        if clsPrm.parametricStudy:
            self.figData = None
            self.Nv = clsPrm.numberPrmStudy
            self.studiedPrmString = None
        else:
            self.figData = libF.FigData(clsPrm,'KS')
            self.Nv = None
        self.sid = None

    # Simulation
    def MonteCarloSimulation(self):
        Ni = self.Ni
        Nc = self.Nc
        p0 = self.p0

        gui = self.progressBar
        Nt = self.Nt
        Ns = self.Ns

        sid = self.sid
        Nv = self.Nv

        l = self.l
        a = self.a
        s = self.s
        f = self.f
        z = self.z
        il = self.il
        typ = self.typ.size

        process = range(Ni)

        shmPrm = {}
        shmHandles = {}
        # clsShM = shmSpecs

        for key in workersShM['parameters'].keys():
            spec, shm = BuildShM(getattr(self,key))
            shmPrm[key] = spec
            shmHandles[key] = shm

        ctx = mp.get_context("spawn")

        if gui: 
            for key,dtype in workersShM['gui'].items():
                spec, shm = BuildShM(Ni=Ni,dtype=dtype)
                shmPrm[key] = spec
                shmHandles[key] = shm

            bar = ProgressBarsGUI(Ni,Nt,sid,Nv,shmPrm,LoadShM)

            try:
                with ProcessPoolExecutor(
                    max_workers=3,
                    mp_context=ctx,
                    initializer=InitializeMCSWorker,
                    initargs=(shmPrm,True)
                ) as executor:
                    futures = {}
                    for p in range(Ni):
                        futures[p] = executor.submit(
                            MonteCarloAlgorithm,
                            p,Nc,Ns,p0,typ,
                            l,a,s,f,z,il,gui
                        )
                    bar.mainloop()

            finally:
                data = [None]*Ni
                for p, fut in futures.items():
                    data[p] = fut.result()
                self.data = data

                for shm in shmHandles.values():
                    shm.close(); shm.unlink()

        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=3,
                    mp_context=ctx,
                    initializer=InitializeMCSWorker,
                    initargs=(shmPrm,)
                ) as executor:
                    data = list(
                        executor.map(
                            MonteCarloAlgorithm,
                            process,[Nc]*Ni,[Ns]*Ni,
                            [p0]*Ni,[typ]*Ni,[l]*Ni,
                            [a]*Ni,[s]*Ni,[f]*Ni,
                            [z]*Ni,[il]*Ni,[gui]*Ni
                        )
                    ); self.data = data
            finally:
                for shm in shmHandles.values():
                    shm.close(); shm.unlink()

        self.EvaluateSimulationData()
        WriteSimulationData(
            self.vrtState,
            self.snapshots,
            self.siVrtState,
            self.typ,
            self.lbl,
            self.li2Name,
            Ni,
            self.sid
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
            
            # This functions applies the mean every ks unit of the Ns snapshots
            return sv

        self.vrtState = np.array([data[p][0] for p in range(Ni)])
        self.snapshots = np.array([data[p][1] for p in range(Ni)])
        self.avrState = Convolve(np.mean(self.snapshots,axis=1))

        self.nodeAvrVrtState = np.mean(self.vrtState,axis=1)
        self.nodeMinVrtState = np.min(self.vrtState,axis=1)
        self.nodeMaxVrtState = np.max(self.vrtState,axis=1)
        self.nodeSumVrtState = np.sum(self.vrtState,axis=1)

        if Ni>1:
            self.ta = stats.t.ppf(0.975,df=Ni-1)

            self.itAvrVrtState = np.mean(self.vrtState,axis=0)
            self.itAvrSnapshots = np.mean(self.snapshots,axis=0)
        else:
            self.ta = None

            self.itAvrVrtState = self.vrtState[0,:,:]
            self.itAvrSnapshots = self.snapshots[0,:,:,:]

        self.itAvrConvSnapshots = Convolve(self.itAvrSnapshots)
        self.siVrtState = np.argsort(self.vrtState,axis=1)
        self.siAvrState = np.argsort(self.itAvrVrtState,axis=0)

    # Figures
    def SizeDistrFittingsFig(
        self,
        ax=None,
        idx=1,
        figData=None,
        saveFig=False
    ):
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

        if figData is None:
            figData = self.figData
            fig, ax = figData.SetFigs(1,3,size=(23,6))
            saveFig = True

        sMax = cs.max(); sMin = cs.min()
        p = (.5,1.07); dp = (.6,.05)

        for t in typ: # t[ype]
            ### Lognormal fit ###
            libF.CreateHistogramPlot(
                cs[:,:,t],30,
                getattr(figData,f'fig{idx}'),
                limits=(sMin,sMax),
                xScale='log',
                Ni=Ni,
                ta=ta,
                label=f"{lbl[t]} histogram",
                color=clr[t],
                alpha=(0.35,0.45) if Ni>1 else 0.35,
                idx=t+idx,
                ax=ax[0]
            ) # Histogram plot

            libF.CreateLognormalFitPlot(
                cs[:,:,t],
                getattr(figData,f'fig{idx}'),
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
                idx=t+idx,
                ax=ax[0]
            ) # Fit plot

            ### Power law fit log-log ###
            bl = libF.CreateParetoFitPlot(
                cs[:,:,t],
                getattr(figData,f'fig{idx+1}'),
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
                idx=t+idx,
                ax=ax[1]
            )

            ### Power law fit log-lin ###
            bl = libF.CreateParetoFitPlot(
                cs[:,:,t],
                getattr(figData,f'fig{idx+2}'),
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
                idx=t+idx,
                ax=ax[2]
            )

            Text(
                ax[1],(p[0],p[1]+(1/2-t)*dp[1]),
                fr'{lbl[t]}:$\quad$'+
                DataString(csMin[:,t],Ni,ta,r's_{{min}}')+
                DataString(csMax[:,t],Ni,ta,r's_{{max}}')+
                DataString(csAvr[:,t],Ni,ta,r'\langle s\rangle')+
                DataString(
                    csSum[:,t],Ni,ta,r's_{{\Sigma}}',
                    formatVal='.2e',formatErr='.2e'
                )+
                DataString(bl,Ni,ta,r'\beta',space=False),
                ha='center',
                color=clr[t]
            )

        if idx == 1:
            p = (0,p[1])
            TextBlock(
                ax[0],[
                    [fr'$Nc={self.Nc}$'],
                    [fr'$R={self.R}$']
                ],
                p=p,dp=(0,dp[1])
            )

            offset = 0 if self.il in (2,4,6) else 1
            p = (.80+.05*offset,p[1])
            dp = (.475-.07*offset,dp[1])
            TextBlock(
                ax[2],[[ '',
                    fr'$\lambda={self.l}$',
                    fr'$\sigma={self.s}$',
                    fr'$Ni={Ni}$',
                    fr'$Ns={self.Ns}$',
                ],[
                    fr'$ζ={self.z}$',
                    '','','',''
                ],[ '',
                    fr'$\alpha={self.a}$',
                    fr'$il={self.il+1}$',
                    fr'$dt={self.dt}$',
                    fr'$sf={self.sf}$',
                ]],
                p=p,dp=dp,
                offset=offset
            )

        if not saveFig: Text(ax[2],(1.1,.5),self.studiedPrmString)

        # Style
        libF.SetFigStyle(
            r'$cs$',r'$P(cs)$',
            yNotation='sci',xScale='log', # ,xNotation="sci"
            ax=ax[0],data=getattr(figData,f'fig{idx}')
        )

        libF.SetFigStyle(
            r'$cs$',r'$P(cs)$',
            xScale='log',yScale='log',
            ax=ax[1],data=getattr(figData,f'fig{idx+1}')
        )

        libF.SetFigStyle(
            r'$cs$',r'$P(cs)$',
            xScale='log',yScale='lin',
            ax=ax[2],data=getattr(figData,f'fig{idx+2}')
        )

        # CentreFig()
        if saveFig: figData.SaveFig('SizeDistributionFittings')

    def AverageSizeFig(
        self,
        ax=None,
        idx=1,
        figData=None,
        saveFig=False
    ):
        Ni = self.Ni
        ta = self.ta

        times = self.times
        ca = self.avrState
        
        lbl = self.lbl
        clr = self.clr

        if figData is None:
            figData = self.figData
            fig = figData.SetFigs()
            saveFig = True

        for t in self.typ:
            libF.CreateFunctionPlot(
                times,
                ca[:,:,t],
                figData.fig if saveFig
                else getattr(figData,f'fig{idx}'),
                Ni=Ni,
                ta=ta,
                label=rf"{lbl[t]} city size average $\langle s\rangle$",
                color=clr[t],
                alpha=(1,0.15) if Ni>1 else 1,
                idx=t+idx,
                ax=ax
            )

        # CentreFig()
        libF.SetFigStyle(
            r"$t$",r"$\langle cs\rangle$",
            data=figData.fig if saveFig
            else getattr(figData,f'fig{idx}'),
            ax=ax
        )

        if saveFig: figData.SaveFig('AverageSize')

    def SizeVsDegreeFig(
        self,
        ax=None,
        idx=1,
        figData=None,
        saveFig=False
    ):
        di = self.di
        cs = self.itAvrVrtState

        typ = self.typ
        lbl = self.lbl
        clr = self.clr

        si = self.siAvrState
        li2Name = self.li2Name

        if figData is None:
            figData = self.figData
            fig, ax = figData.SetFigs(1,2,size=(15,6))
            saveFig = True

        for i,scale in enumerate(['lin','log']):
            for t in typ:
                libF.CreateScatterPlot(
                    di,cs[:,t],
                    getattr(figData,f'fig{i+idx}'),
                    label=lbl[t],
                    color=clr[t],
                    idx=t+idx,
                    ax=ax[i]
                )

            libF.SetFigStyle(
                r'$k$',r'$cs(k)$',
                yScale='log',xScale=scale,
                data=getattr(figData,f'fig{i+idx}'),
                ax=ax[i]
            )

        for t in typ:
            TextBlock(
                ax[0],[[
                    DataString(
                        cs[si[i,t],t],
                        head=li2Name[si[i,t]],
                        formatVal='.2e',
                        space=False
                    )
                ] for i in range(-1,-6,-1)],
                p=(.8,.35-t*.225),
                dp=(0,.15),
                ha='center',
                color=clr[t]
            )

        # CentreFig()
        if saveFig: figData.SaveFig('SizeVsDegree')

    def SizeDistrEvolutionFig(
        self,
        ax=None,
        idx=1,
        figData=None,
        saveFig=False
    ):
        snapshots = self.itAvrSnapshots

        ns = self.ns
        Ns = self.Ns

        dt = self.dt
        typ = self.typ

        if figData is None:
            figData = self.figData
            fig, ax = figData.SetFigs(1,2,size=(15,6))
            saveFig = True
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')

        samples = 6
        clrmap = get_cmap('inferno') # magma
        colours = [clrmap(i/(samples-1)) for i in range(samples)]

        sMax = np.max(snapshots[:,-1,:])
        sMin = np.min(snapshots[:,-1,:])

        for t in typ: # t[ype]
            for j,s in enumerate(np.linspace(0,Ns,samples,dtype=int)):
                libF.CreateHistogramPlot(
                    snapshots[:,s,t],21,
                    getattr(figData,f'fig{t+idx}'),
                    limits=(sMin,sMax),
                    xScale='log',
                    label=f"t = {int(ns[s]*dt)}",
                    color=colours[j][:-1],
                    alpha=0.4,
                    idx=j+idx,
                    ax=ax[t],
                    norm=False
                ) # Histogram plot

            libF.SetFigStyle(
                r"$cs$",r"$P(cs)$",
                yNotation="sci", # ,xNotation="sci"
                xScale='log',
                yScale='log',
                ax=ax[t],
                data=getattr(figData,f'fig{t+idx}')
            ) # Style

        # CentreFig()
        if saveFig: figData.SaveFig('SizeDistributionEvolution')

    def SizeEvolutionsFig(
        self,
        ax=None,
        idx=1,
        figData=None,
        saveFig=False
    ):
        # Nc = self.Nc
        typ = self.typ

        times = self.times
        snapshots = self.itAvrConvSnapshots

        di = self.di
        sf = self.sf

        if figData is None:
            figData = self.figData
            fig, ax = figData.SetFigs(1,2,size=(15,6))
            saveFig = True
        ax[0].set_title('Exact'); ax[1].set_title('Approximated')

        dk, _ = np.unique(di,return_counts=True); Nk = dk.size
        snapshotsk = np.zeros((Nk,sf+1,2),dtype=np.float64)
        for i,k in enumerate(dk):
            bk = di == k
            snapshotsk[i,:,:] = snapshots[bk,:,:].mean(axis=0)

        clrmap = get_cmap('inferno') # magma
        norm = LogNorm(vmin=dk.min(),vmax=dk.max())
        colours = clrmap(norm(dk[::-1]))
        # colours = [clrmap(i/(Nc-1)) for i in range(Nc)]

        labels = ['']*Nk
        classes = np.linspace(0,Nk-1,6,dtype=np.int64)
        for i in classes: labels[i] = f'k={dk[i]}'

        for t in typ: # t[ype]
            # norm = LogNorm(
            #     vmin=snapshotsk[:,-1,t].min(),
            #     vmax=snapshotsk[:,-1,t].max()
            # )
            # colours = clrmap(norm(snapshotsk[:,-1,t]))

            for k in np.linspace(Nk-1,0,Nk,dtype=np.int64):
                libF.CreateFunctionPlot(
                    times,snapshotsk[k,:,t],
                    getattr(figData,f'fig{t+idx}'),
                    color=colours[k,:],
                    alpha=0.8,
                    linewidth=1,
                    label=labels[k],
                    idx=k+idx,
                    ax=ax[t]
                ) # Histogram plot

            libF.SetFigStyle(
                r"$t$",r"$s$",
                # yNotation="sci", # ,xNotation="sci"
                yScale='log',
                data=getattr(figData,f'fig{t+idx}'),
                ax=ax[t]
            ) # Style

        # CentreFig()
        if saveFig: figData.SaveFig('SizeEvolutions')

    def ShowFig(self):
        from matplotlib.pyplot import show
        show()

class ParametricStudy():
    def __init__(self,clsPrm,clsReg):
        sp = clsPrm.studiedParameter

        sV = clsPrm.startValuePrmStudy; self.sV = sV
        eV = clsPrm.endValuePrmStudy; self.eV = eV
        Nv = clsPrm.numberPrmStudy; self.Nv = Nv

        self.KS = [None]*Nv
        for s,val in enumerate(np.linspace(sV,eV,Nv)):
            match sp:
                case 0: 
                    clsPrm.attractivity = val
                    string = fr'$\lambda={val:.2f}$'
                case 1:
                    clsPrm.convincibility = val
                    string = fr'$\alpha={val:.2f}$'
                case 2:
                    clsPrm.zetaFraction = val
                    string = fr'$\zeta={val:.2f}$'

            self.KS[s] = KineticSimulation(clsPrm,clsReg)
            self.KS[s].sid = s+1
            self.KS[s].studiedPrmString = string 

        self.fw = 24
        self.fh = 8
        self.figData = libF.FigData(clsPrm,'KS')

    # Simulation
    def MonteCarloSimulation(self):
        for sim in self.KS:
            sim.MonteCarloSimulation()

    # Figures
    def SizeDistrFittingsFig(self):
        figData = self.figData
        fw = self.fw
        fh = self.fh

        nCol = 3 # Figures for each row
        nRow = self.Nv
        fig, ax = figData.SetFigs(nRow,nCol,size=(fw,fh*nRow))

        for s,KS in enumerate(self.KS):
            KS.SizeDistrFittingsFig(
                ax=ax[s,:],
                idx=nCol*s+1,
                figData=figData
            )

        figData.SaveFig('SizeDistributionFittings')

    def AverageSizeFig(self):
        figData = self.figData
        fw = self.fw/3
        fh = self.fh

        nCol = 1 # Figures for each row
        nRow = self.Nv
        fig, ax = figData.SetFigs(nRow,nCol,size=(fw,fh*nRow))

        for s,KS in enumerate(self.KS):
            KS.AverageSizeFig(
                ax=ax[s],
                idx=nCol*s+1,
                figData=figData
            )

        figData.SaveFig('AverageSize')

    def SizeVsDegreeFig(self):
        figData = self.figData
        fw = self.fw*2/3
        fh = self.fh

        nCol = 2 # Figures for each row
        nRow = self.Nv
        fig, ax = figData.SetFigs(nRow,nCol,size=(fw,fh*nRow))

        for s,KS in enumerate(self.KS):
            KS.SizeVsDegreeFig(
                ax=ax[s,:],
                idx=nCol*s+1,
                figData=figData
            )

        figData.SaveFig('SizeVsDegree')

    def SizeDistrEvolutionFig(self):
        figData = self.figData
        fw = self.fw*2/3
        fh = self.fh

        nCol = 2 # Figures for each row
        nRow = self.Nv
        fig, ax = figData.SetFigs(nRow,nCol,size=(fw,fh*nRow))

        for s,KS in enumerate(self.KS):
            KS.SizeDistrEvolutionFig(
                ax=ax[s,:],
                idx=nCol*s+1,
                figData=figData
            )

        figData.SaveFig('SizeDistributionEvolution')

    def SizeEvolutionsFig(self):
        figData = self.figData
        fw = self.fw*2/3
        fh = self.fh

        nCol = 2 # Figures for each row
        nRow = self.Nv
        fig, ax = figData.SetFigs(nRow,nCol,size=(fw,fh*nRow))

        for s,KS in enumerate(self.KS):
            KS.SizeEvolutionsFig(
                ax=ax[s,:],
                idx=nCol*s+1,
                figData=figData
            )

        figData.SaveFig('SizeEvolutions')

    def ShowFig(self):
        from matplotlib.pyplot import show
        show()


### Auxiliary functions ###

@dataclass(frozen=True)
class ShMSpec:
    name: str
    shape: tuple[int,...]
    dtype: str

def BuildShM(
    array=None,
    Ni=None,
    dtype=None
):
    if array is not None:
        a = np.ascontiguousarray(array)
    else:
        a = np.zeros(Ni,dtype=dtype)

    shm = shared_memory.SharedMemory(
        create=True,
        size=a.nbytes
    )

    view = np.ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=shm.buf
    )
    view[:] = a

    spec = ShMSpec(
        name=shm.name,
        shape=a.shape,
        dtype=a.dtype.str
    )

    return spec, shm

def LoadShM(spec):
    shm = shared_memory.SharedMemory(name=spec.name)
    arr = np.ndarray(
        spec.shape,
        dtype=np.dtype(spec.dtype),
        buffer=shm.buf
    )
    return shm, arr

def InitializeMCSWorker(
    shmPrm,
    gui=False
):
    for key in workersShM['parameters'].keys():
        shm, arr = LoadShM(shmPrm[key])
        workersShM['handles'].append(shm)
        workersShM['parameters'][key] = arr

    if gui:
        for key in workersShM['gui'].keys():
            shm, arr = LoadShM(shmPrm[key])
            workersShM['handles'].append(shm)
            workersShM['gui'][key] = arr

def MonteCarloAlgorithm(
    p,Nc,Ns,p0,typ,
    l,a,s,f,z,il,gui
):
    Mdt = workersShM['parameters']["Mdt"]
    wOdt = workersShM['parameters']["wOdt"]
    wI = workersShM['parameters']["wI"]
    di = workersShM['parameters']["di"]
    idi = workersShM['parameters']["idi"]
    ns = workersShM['parameters']["ns"]

    # rng = np.random.default_rng() # Even though it's recommended by Numpy it is not efficiently implemented in Numba, hence it halves the iterations per second if used

    # Uniform initial state for all vertices
    vrtState = np.full((Nc,typ),p0,dtype=np.float64)
    snapshots = np.full((Nc,Ns+1,typ),p0,dtype=np.float64)

    P = np.arange(Nc,dtype=np.int32)

    nk = ns[1]
    hNc = Nc//2 # Nc half
    nsid = 1

    if gui: # If ProgressGUI is required
        progress = workersShM['gui']['progress']
        elapsed = workersShM['gui']['elapsed']
        done = workersShM['gui']['done']

        EvolveState(
            vrtState,P,
            Nc,hNc,nk,
            Mdt,wOdt,wI,
            di,idi,il,
            l,a,s,f,z#,rng
        ) # Warm-up iteration to avoid polluting the initial time t0
        snapshots[:,nsid,0] = vrtState[:,0]
        snapshots[:,nsid,1] = vrtState[:,1]
        nsid += 1

        t0 = time.perf_counter()
        for st in range(Ns-1): # step
            EvolveState(
                vrtState,P,
                Nc,hNc,nk,
                Mdt,wOdt,wI,
                di,idi,il,
                l,a,s,f,z#,rng
            )
            snapshots[:,nsid,0] = vrtState[:,0]
            snapshots[:,nsid,1] = vrtState[:,1]
            nsid += 1

            progress[p] = (st+2)*nk # ns[nsid]
            elapsed[p] = time.perf_counter()-t0

        progress[p] = Ns*nk # ns[-1]
        elapsed[p] = time.perf_counter()-t0
        done[p] = True

        return vrtState, snapshots

    else:
        EvolveState(
            vrtState,P,
            Nc,hNc,nk,
            Mdt,wOdt,wI,
            di,idi,il,
            l,a,s,f,z#,rng
        ) # Warm-up iteration to avoid polluting the initial time t0
        snapshots[:,nsid,0] = vrtState[:,0]
        snapshots[:,nsid,1] = vrtState[:,1]
        nsid += 1

        t0 = time.perf_counter()
        for _ in range(Ns-1):
            EvolveState(
                vrtState,P,
                Nc,hNc,nk,
                Mdt,wOdt,wI,
                di,idi,il,
                l,a,s,f,z#,rng
            )
            snapshots[:,nsid,0] = vrtState[:,0]
            snapshots[:,nsid,1] = vrtState[:,1]
            nsid += 1

        return vrtState, snapshots

@njit(cache=True)
def EvolveState(
    cs,P,Nc,hNc,nk,
    Mdt,wOdt,wI,
    di,idi,il,
    l,a,s,f,z#,rng
):
    for _ in range(nk):
        FYDInPlaceShuffle(P,Nc)
        # P = np.random.permutation(Nc)
        # pi = P[:hNc]; pr = P[hNc:]

        for i in range(hNc):
            ii = P[i]; ir = P[i+hNc]

            # t = 0
            p = Mdt[ii,ir]
            if p > 0:
                theta = np.random.random() < p

                if theta == 1:
                    si = cs[ii,0]; sr = cs[ir,0]

                    e = NonLinearEmigration(si,idi[ii],sr,di[ir],il,l,a,z)
                    ga = StochasticFluctuations(s,e) if f else 0

                    cs[ii,0] = si*(1-e+ga) 
                    cs[ir,0] = sr+si*e

            # t = 1
            p = wOdt[ii]*wI[ir] 
            # Apparently it's more efficient to access two values from two separate vectors and to multiply them, than it is to access the same pre-computed value from a matrix; the same holds for the product «di[ir]*idi[ii]» inside «NonLinearEmigration()»
            # However, in order for this property to be valid the matrix has to have an underlying more simple structure which in both cases is rank 1
            # In other words this trick does not work with the adjacency matrix «Mdt[ii,ir]» which cannot be computed from simpler elements
            theta = np.random.random() < p

            if theta == 1:
                si = cs[ii,1]; sr = cs[ir,1]

                e = NonLinearEmigration(si,idi[ii],sr,di[ir],il,l,a,z)
                ga = StochasticFluctuations(s,e) if f else 0

                cs[ii,1] = si*(1-e+ga) 
                cs[ir,1] = sr+si*e

            # In the exact case it's reasonalbe to check whether p is actually positive before evaluating the Bernoulli distribution with «np.random.random()<p» since A can has zero components
            # However its approximations Ap does not have zero components by definition (it's a complete network), hence that check becomes useless

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
        si,idii, # Interacting city size
        sr,dir,  # Receiving city size
        il,l,a,z
    ):
    if si <= 0: return 0

    if il == 0:
        rs = sr/si
        return l*(rs**a)/(1+rs**a)

    elif il == 1:
        rsk = (sr/si)*dir*idii
        return l*(rsk/a)/(1+rsk/a)

    elif il == 2:
        if sr <= 0: return 0

        rsk = (sr/si)*dir*idii
        efl = l*(rsk/a)/(1+rsk/a)

        irsk = 1/rsk
        efs = l*(irsk/a)/(1+irsk/a)

        return (1-z)*efl+z*efs

    elif il == 3:
        rsk = (sr/si)*dir*idii
        return l*(rsk**a)/(1+rsk**a)

    elif il == 4:
        if sr <= 0: return 0

        rsk = (sr/si)*dir*idii
        efl = l*(rsk**a)/(1+rsk**a)

        irsk = 1/rsk
        efs = l*(irsk**a)/(1+irsk**a)

        return (1-z)*efl+z*efs

    elif il == 5:
        rsk = (sr/si)*dir*idii
        return l*(rsk/(1+rsk))**a

    else:
        if sr <= 0: return 0

        rsk = (sr/si)*dir*idii
        efl = l*(rsk/(1+rsk))**a

        irsk = 1/rsk
        efs = l*(irsk/(1+irsk))**a

        return (1-z)*efl+z*efs

        # rsk = (sr/si)*(dr*idi)  # Relative population*degree ratio
        # irsk = (si/sr)*(di*idr) # Inverse relative population*degree ratio
        # efl                     # Actual emigration rate for the lumping fraction
        # efs                     # Actual emigration rate for the separation fraction

    # Numba does not officially support the match structure, hence, even if it appears to work, it's better to use an «if/elif/else» one the sake of performance and stability

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

def Text(
    target,
    pos,
    string,
    ha='center',
    color=None
):
    if hasattr(target,"transAxes"):
        target.text(
            pos[0],
            pos[1],
            string,
            color=color,
            ha=ha,
            transform=target.transAxes
        )
    else:
        target.text(
            pos[0],
            pos[1],
            string,
            color=color,
            ha=ha
        )

def TextBlock(
    ax,
    list,
    p=(0,0),
    dp=(0,0),
    offset=0,
    **kwargs
):
    nR = len(list); nC = len(list[0])-offset
    for r,y in enumerate(np.linspace(p[1]+dp[1]/2,p[1]-dp[1]/2,nR)):
        for c,x in enumerate(
            np.linspace(p[0]-dp[0]/2,p[0]+dp[0]/2,nC),
            start=offset
        ):
            Text(ax,(x,y),list[r][c],**kwargs)
            # x moves from left to right
            # y moves from top to bottom

    # An alternative is to do what linspace does manually with «x0+(i-(n-1)/2)*dx» where dx is actually the space between strings rather than the block length


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
