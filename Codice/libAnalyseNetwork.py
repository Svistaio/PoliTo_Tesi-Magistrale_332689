
import numpy as np
from scipy.stats import lognorm, linregress
import networkx as nx

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import mpld3

import importlib
import libFigures as libF
importlib.reload(libF)


### Main class ###

class NetworkAnalysis:
    def __init__(self,dicReg):
        self.A = dicReg['A']
        self.W = dicReg['W']
        self.n = dicReg['Nc']

        # Degrees
        self.di = np.sum(self.A,axis=0) # Vectors of degrees
        self.dk, self.Nk = np.unique(self.di,return_counts=True)
        # Unique degrees and corresponding frequencies
        # Pk = counts/N

        # [Nonzero] Weights
        self.wi = self.W[self.W>0]
        self.wk, self.wNk = np.unique(self.wi,return_counts=True)
        # Unique weights and corresponding frequencies

        # Strenghts
        self.si = np.sum(self.W,axis=0)
        self.sk, self.sNk = np.unique(self.si,return_counts=True)
        # Unique strenghts and corresponding frequencies

    def DegreeDistributionFig(self):
        di = self.di

        fig = plt.figure()
        dicData = libF.CreateDicData(1)

        kAvr = np.mean(di)
        fig.text(
            0.8,0.25,
            fr"$N$={di.size}"f"\n"
            fr"$E$={int(np.sum(di)/2)}"f"\n"
            fr"$k_{{min}}$={np.min(di)}"f"\n"
            fr"$k_{{max}}$={np.max(di)}"f"\n"
            r"$\langle k\rangle $="f"{kAvr}",
            ha="center",
            # fontsize=10,
            color="black"
        )


        # Histogram plot
        libF.CreateHistogramPlot(di,25,dicData['fig'])

        #region
            # Scatter plot
            # scP = plt.scatter(k,Pk,label="Empirical",s=10)
        
            # mplcursors.cursor(scP,hover=False).connect(
            #     "add",lambda sel: sel.annotation.set_text(
            #         f"k={k[sel.index]}, P(k)={Pk[sel.index]:.3f}"
            #     )
            # )
        
        
            # # Manual fitting (ML)
        
            # # Estimators
            # m = (1/N)*np.sum(np.log(d)) # Estimated mean
            # s2 = (1/N)*np.sum((np.log(d)-m)**2)
            # s = np.sqrt(s2) # Estimated standard deviation
        
            # # Evaluation
            # def lognormalPDF(x,m,s):
            #     y = np.zeros_like(x)
            #     v = x>0
            #     C = 1/(x[v]*s*np.sqrt(2*np.pi))
            #     y[v] = C*np.exp(-(np.log(x[v])-m)**2/(2*s**2))
            #     return y
        
            # # Plotting
            # scF = plt.plot(
            #     x,lognormalPDF(x,m,s),
            #     label="Manual lognormal fit (ML)", # Maximum likelyhood
            #     color="red",
            #     linewidth=1
            # )
        #endregion


        # SciPy fitting (ML)
        libF.CreateLognormalFitPlot(di,dicData['fig'])


        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$P(k)$",
            [0,300],[0,0.03],
            data=dicData['fig']
        )
        libF.SaveFig('DegreeDistribution','NA',dicData)

    def WeightDistributionFig(self):
        wi = self.wi

        fig = plt.figure()
        dicData = libF.CreateDicData(1)


        # Histogram plot
        hgPlot = libF.CreateHistogramPlot(wi,20,dicData['fig'],scale='log')


        # SciPy regression
        binw = (hgPlot[1][1:]+hgPlot[1][:-1])/2
        binPw = hgPlot[0]
        # sc = plt.scatter(binw,binPw,c='r',s=50)

        # Fit in log–log space
        slope = libF.CreateLogRegressionPlot(binw,binPw,dicData['fig'])

        kAvr = np.mean(wi)
        fig.text(
            0.75,0.5,
            r"$w_{min}$="f"{np.min(wi)}\n"
            r"$w_{max}$="f"{np.max(wi)}\n"
            r"$\langle w\rangle $="f"{kAvr:.3f}\n"
            r"$\alpha$="f"{slope:.3f}",
            ha="center",
            # fontsize=10,
            color="black"
        )


        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$w$",r"$P(w)$",
            xScale="log",yScale="log",
            data=dicData['fig']
        )
        libF.SaveFig('WeightDistribution','NA',dicData)

    def StrengthDistributionFig(self):
        si = self.si

        fig = plt.figure()
        dicData = libF.CreateDicData(1)


        # Histogram plot
        hgPlot = libF.CreateHistogramPlot(si,20,dicData['fig'],scale='log')


        # SciPy regression
        bins = (hgPlot[1][1:]+hgPlot[1][:-1])/2
        binPs = hgPlot[0]
        # sc = plt.scatter(binw,binPw,c='r',s=50)

        # Fit in log–log space
        v = binPs>0; v[:6] = 0
        slope = libF.CreateLogRegressionPlot(bins[v],binPs[v],dicData['fig'])

        kAvr = np.mean(si)
        fig.text(
            0.75,0.5,
            fr"$s_{{min}}$={np.min(si)}"f"\n"
            fr"$s_{{max}}$={np.max(si)}"f"\n"
            fr"$\langle s\rangle $={kAvr:.3f}"f"\n"
            fr"$\alpha$={slope:.3f}",
            ha="center",
            # fontsize=10,
            color="black"
        )


        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$s$",r"$P(s)$",
            xScale="log",yScale="log",
            data=dicData['fig']
        )
        libF.SaveFig('StrengthDistribution','NA',dicData)

    def BetweennessCentralityFig(self):
        A = self.A; di = self.di

        fig = plt.figure()
        dicData = libF.CreateDicData(1)
        
        G = nx.from_numpy_array(A)
        bc = nx.betweenness_centrality(G,normalized=False)
        bcd = np.array([bc[i] for i in range(len(di))])

        aAvr = np.mean(list(bc.values()))
        fig.text(
            0.75,0.25,
            r"$g_{min}$="+f"{np.min(list(bc.values())):.3f}"+"\n"+\
            r"$g_{max}$="+f"{np.max(list(bc.values())):.0e}"+"\n"+\
            r"$\langle g\rangle $="+f"{aAvr:.3f}",
            ha="center",
            # fontsize=10,
            color="black"
        )

        libF.CreateScatterPlot(di,bcd,dicData['fig'],l='')


        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$g(i)$",
            # [0.5e1,0.5e3],[1,0.5e5],
            xScale='log',yScale='log',
            data=dicData['fig']
        )
        libF.SaveFig('BetweennessCentrality','NA',dicData)

    def StrengthVsDegreeFig(self):
        si = self.si; di = self.di
        dk = self.dk; Nk = self.Nk

        fig = plt.figure()
        dicData = libF.CreateDicData(1)


        # Scatter
        sk = np.zeros_like(dk,dtype=float)
        for i,ki in enumerate(dk):
            v = np.nonzero(di == ki)[0]
            sk[i] = np.sum(si[v])
            sk[i] /= Nk[i]

        libF.CreateScatterPlot(dk,sk,dicData['fig'])

        # Fit in log–log space
        slope = libF.CreateLogRegressionPlot(dk,sk,dicData['fig'])

        sAvr = np.mean(si)
        fig.text(
            0.75,0.25,
            r"$s_{min}$="f"{np.min(si)}\n"
            r"$s_{max}$="f"{np.max(si)}\n"
            r"$\langle s\rangle $="f"{sAvr:.3f}\n"
            r"$\alpha$="f"{slope:.3f}",
            ha="center",
            # fontsize=10,
            color="black"
        )


        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$s(k)$",
            xScale="log",yScale="log",
            data=dicData['fig']
        )
        libF.SaveFig('StrengthVsDegree','NA',dicData)

    def AClusteringCoefficientFig(self):
        A = self.A; di = self.di
        dk = self.dk; Nk = self.Nk

        fig = plt.figure()
        dicData = libF.CreateDicData(1)
        
        G = nx.from_numpy_array(A)
        Cd = nx.clustering(G)

        Ck = np.zeros_like(dk,dtype=float)
        for i,ki in enumerate(dk):
            v = np.nonzero(di == ki)[0]
            for n in v:
                Ck[i] += Cd[n]
            Ck[i] /= Nk[i]
        self.Ck = Ck

        #region
            # # Manual counting
            # Cd = np.zeros_like(d,dtype=float)
            # for i,ki in enumerate(d):
            #     if ki>1: # Consider only nodes with more than 2 neighbours
            #         vi = A[i,:]
            #         indeces = np.nonzero(vi)[0]

            #         Ei = 0
            #         for n in indeces:
            #             for m in indeces:
            #                 if A[n,m] == 1:
            #                     Ei += 1
            #         Ei /= 2 # Divided by 2 since each edge is counted twice

            #         Cd[i] = 2*Ei/(ki*(ki-1))
            #     else:
            #         Cd[i] = 0

            # Ck = np.zeros_like(k,dtype=float)
            # for i,ki in enumerate(k):
            #     v = (d == ki)
            #     Nk = np.sum(v)

            #     Ck[i] = np.sum(Cd[v])/Nk
        #endregion

        CAvr = nx.average_clustering(G)
        fig.text(
            0.75,0.75,
            fr"$C_{{min}}$={np.min(list(Cd.values())):.3f}"f"\n"
            fr"$C_{{max}}$={np.max(list(Cd.values())):.3f}"f"\n"
            fr"$\langle C\rangle $={CAvr:.3f}",
            ha='center',
            # fontsize=10,
            color="black"
        )

        libF.CreateScatterPlot(dk,Ck,dicData['fig'],l='')

        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$C(k)$",
            [0,300],[0,0.8],
            data=dicData['fig']
        )
        libF.SaveFig('AClusteringCoefficient','NA',dicData)

    def WClusteringCoefficientFig(self):
        A = self.A; W = self.W;
        si = self.si; di = self.di; dk = self.dk
        Nk = self.Nk; Ck = self.Ck

        fig, ax = plt.subplots(2,1)
        dicData = libF.CreateDicData(2)


        # Manual counting
        Cd = np.zeros_like(di,dtype=float)
        for i, ki in enumerate(di):
            if ki>1: # Consider only nodes with more than 2 neighbours
                vi = A[i,:]
                indeces = np.nonzero(vi)[0]

                Ei = 0
                for n in indeces:
                    for m in indeces:
                        Ei += (W[i,n]+W[i,m])*A[n,m]
                Ei /= 2 # Divided by 2 since each edge is counted twice

                Cd[i] = Ei/(si[i]*(ki-1))
            else:
                Cd[i] = 0


        Ckw = np.zeros_like(dk,dtype=float)
        for i, ki in enumerate(dk):
            v = (di == ki)
            Ckw[i] = np.sum(Cd[v])/Nk[i]

        libF.CreateScatterPlot(dk,Ckw,dicData['fig1'],l='',ax=ax[0])

        # Style
        libF.SetFigStyle(
            r"$k$",r"$C^w(k)$",
            xScale="log",ax=ax[0],
            data=dicData['fig1']
        )


        CkwRel= (Ckw-Ck)/Ck
        libF.CreateScatterPlot(dk,CkwRel,dicData['fig2'],l='',ax=ax[1])
        self.CkwRel = CkwRel

        # Style
        libF.SetFigStyle(
            r"$k$",r"$C^w_{\text{rel}}(k)$",
            xScale="log",yScale="log",
            ax=ax[1],data=dicData['fig2']
        )


        libF.CentreFig()
        libF.SaveFig('WClusteringCoefficient','NA',dicData)

    def AAssortativityFig(self):
        A = self.A; dk = self.dk

        fig = plt.figure()
        dicData = libF.CreateDicData(1)
        
        G = nx.from_numpy_array(A)
        ad = nx.average_neighbor_degree(G)
        ak = nx.average_degree_connectivity(G)

        aAvr = np.mean(list(ad.values()))
        fig.text(
            0.75,0.75,
            r"$k_{nn}^{min}$="+f"{np.min(list(ad.values())):.3f}"+"\n"+\
            r"$k_{nn}^{max}$="+f"{np.max(list(ad.values())):.3f}"+"\n"+\
            r"$\langle k_{nn}\rangle $="+f"{aAvr:.3f}",
            ha="center",
            # fontsize=10,
            color="black"
        )

        knn = np.array([ak[ki] for ki in dk])
        libF.CreateScatterPlot(dk,knn,dicData['fig'],l='')
        self.knn = knn


        # Style
        libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$k_{nn}(k)$",
            [0,300],[40,95],
            data=dicData['fig']
        )
        libF.SaveFig('AAssortativity','NA',dicData)

    def WAssortativityFig(self):
        W = self.W; dk = self.dk; knn = self.knn

        fig, ax = plt.subplots(2,1)
        dicData = libF.CreateDicData(2)

        Gw = nx.from_numpy_array(W)
        # adw = nx.average_neighbor_degree(Gw,weight="weight")
        akw = nx.average_degree_connectivity(Gw,weight="weight")


        knnw = np.array([akw[ki] for ki in dk])
        libF.CreateScatterPlot(dk,knnw,dicData['fig1'],l='',ax=ax[0])

        # Style
        libF.SetFigStyle(
            r"$k$",r"$k_{nn}^w(k)$",
            xScale="log",yScale="log",
            ax=ax[0],data=dicData['fig1']
        )


        knnwRel = (knnw-knn)/knn
        libF.CreateScatterPlot(dk,knnwRel,dicData['fig2'],l='',ax=ax[1])

        # Style
        libF.SetFigStyle(
            r"$k$",r"$k_{nn,rel}^w(k)$",
            xScale="log",yScale="log",
            ax=ax[1],data=dicData['fig2']
        )


        libF.CentreFig()
        libF.SaveFig('WAssortativity','NA',dicData)

    def ShowFig(self):
        plt.show()
        mpld3.show()
