
# Library to analyse networks [also known as graphs]

import numpy as np
import networkx as nx

import libFigures as libF


### Main class ###

class NetworkAnalysis():
    def __init__(self,clsPrm,clsReg):
        self.li2Name = clsReg.li2Name
        self.name2li = clsReg.name2li
        self.Nc = clsReg.Nc

        A = clsReg.A; self.A = A
        W = clsReg.W; self.W = W

        # Degrees
        di = np.sum(A,axis=0); self.di = di # Degree vector
        self.dk, self.Nk = np.unique(di,return_counts=True)
        # Unique degrees and corresponding frequencies
        # Pk = counts/N

        # [Nonzero] Weights
        wi = W[W>0]; self.wi = wi
        self.wk, self.wNk = np.unique(wi,return_counts=True)
        # Unique weights and corresponding frequencies

        # Strenghts
        si = np.sum(W,axis=0); self.si = si
        self.sk, self.sNk = np.unique(si,return_counts=True)
        # Unique strenghts and corresponding frequencies

        self.figData = libF.FigData(clsPrm,'NA')

    def DegreeDistributionFig(self):
        li2Name = self.li2Name
        di = self.di

        figData = self.figData
        fig = figData.SetFigs()


        kAvr = np.mean(di)
        libF.TextBlock(
            fig,[
                [libF.DataString(di.size,head='N',space=False,formatVal='')],
                [libF.DataString(int(np.sum(di)/2),head='E',space=False,formatVal='')],
                [libF.DataString(np.min(di),head=r'k_{{min}}',space=False,formatVal='')],
                [libF.DataString(np.max(di),head=r'k_{{max}}',space=False,formatVal='')],
                [libF.DataString(kAvr,head=r'\langle k\rangle',space=False)]
            ],(0.8,0.4),
            (0,0.15)
        )


        kis = np.argsort(di) # Vector for the sorted degrees
        kit = [['City','k']] # Table for the sorted degrees
        for i in range(10):
            li = kis[-i-1]
            kit.append([li2Name[li],di[li]])
        libF.TextBlock(fig,kit,(1.175,0.5),(0.2,0.4))


        # Histogram plot
        libF.CreateHistogramPlot(di,25,figData.fig)

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
        libF.CreateLognormalFitPlot(di,figData.fig)


        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$P(k)$",
            # [0,300],[0,0.03],
            data=figData.fig
        )
        figData.SaveFig('DegreeDistribution')

    def WeightDistributionFig(self):
        W = self.W
        wi = self.wi
        Nc = self.Nc

        li2Name = self.li2Name
        name2li = self.name2li

        figData = self.figData
        fig = figData.SetFigs()


        # Histogram plot
        binw, binPw = libF.CreateHistogramPlot(wi,19,figData.fig,xScale='log')


        # SciPy regression
        # binw = (hgPlot[1][1:]+hgPlot[1][:-1])/2
        # binPw = hgPlot[0]
        # sc = plt.scatter(binw,binPw,c='r',s=50)

        # Fit in log–log space
        slope = libF.CreateLogRegressionPlot(binw,binPw,figData.fig)

        kAvr = np.mean(wi)
        libF.TextBlock(
            fig,[
                [libF.DataString(np.min(wi),head=r'w_{min}',space=False,formatVal='')],
                [libF.DataString(np.max(wi),head=r'w_{max}',space=False,formatVal='')],
                [libF.DataString(kAvr,head=r'\langle w\rangle',space=False,formatVal='.3f')],
                [libF.DataString(slope,head=r'\alpha',space=False,formatVal='.3f')]
            ],(0.75,0.5),
            (0,0.1)
        )


        iWu,jWu = np.triu_indices_from(W,k=1)
        values = W[iWu,jWu]
        ls = np.argsort(values)[-5:][::-1] # Vector or sorted links

        Wt = [['Connection [extracted]', 'w']]
        for l in ls:
            lir = iWu[l]; lic = jWu[l]
            Wt.append([f'{li2Name[lir]}-{li2Name[lic]}',W[lir,lic]])
        libF.TextBlock(fig,Wt,(1.3,0.65),(0.3,0.2))


        Wt = [['Connection [reference]','w']]
        for link in [
            ['CAGLIARI','SASSARI'],
            ['SASSARI','OLBIA'],
            ['CAGLIARI','ASSEMINI'],
            ['PORTO TORRES','SASSARI'],
            ['CAGLIARI','CAPOTERRA']
        ]:
            lir = name2li[link[0]]
            lic = name2li[link[1]]
            Wt.append([f'{li2Name[lir]}-{li2Name[lic]}',W[lir,lic]])
        libF.TextBlock(fig,Wt,(1.3,0.35),(0.3,0.2))


        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$w$",r"$P(w)$",
            xScale="log",yScale="log",
            data=figData.fig
        )
        figData.SaveFig('WeightDistribution')

    def StrengthDistributionFig(self):
        li2Name = self.li2Name
        si = self.si

        figData = self.figData
        fig = figData.SetFigs()


        # Histogram plot
        bins, binPs = libF.CreateHistogramPlot(si,20,figData.fig,xScale='log')


        # SciPy regression
        # bins = (hgPlot[1][1:]+hgPlot[1][:-1])/2
        # binPs = hgPlot[0]
        # sc = plt.scatter(binw,binPw,c='r',s=50)

        # Fit in log–log space
        v = binPs>0; v[:6] = 0
        slope = libF.CreateLogRegressionPlot(bins[v],binPs[v],figData.fig)


        kAvr = np.mean(si)
        libF.TextBlock(
            fig,[
                [libF.DataString(np.min(si),head=r's_{min}',space=False,formatVal='')],
                [libF.DataString(np.max(si),head=r's_{max}',space=False,formatVal='')],
                [libF.DataString(kAvr,head=r'\langle s\rangle',space=False,formatVal='.3f')],
                [libF.DataString(slope,head=r'\alpha',space=False,formatVal='.3f')]
            ],(0.75,0.5),
            (0,0.1)
        )

        
        sis = np.argsort(si) # Vector for the sorted strengths
        sit = [['City','s']] # Table for the sorted strengths
        for i in range(10):
            li = sis[-i-1]
            sit.append([li2Name[li],si[li]])
        libF.TextBlock(fig,sit,(1.175,0.5),(0.2,0.4))


        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$s$",r"$P(s)$",
            xScale="log",yScale="log",
            data=figData.fig
        )
        figData.SaveFig('StrengthDistribution')

    def BetweennessCentralityFig(self):
        A = self.A
        di = self.di

        figData = self.figData
        fig = figData.SetFigs()
        
        G = nx.from_numpy_array(A)
        bc = nx.betweenness_centrality(G,normalized=False)
        bcd = np.array([bc[i] for i in range(len(di))])

        aAvr = np.mean(list(bc.values()))
        libF.TextBlock(
            fig,[
                [libF.DataString(np.min(list(bc.values())),head=r'g_{min}',space=False,formatVal='.3f')],
                [libF.DataString(np.max(list(bc.values())),head=r'g_{max}',space=False,formatVal='.0f')],
                [libF.DataString(aAvr,head=r'\langle g\rangle',space=False,formatVal='.3f')]
            ],(0.75,0.25),
            (0,0.075)
        )

        libF.CreateScatterPlot(
            di,bcd,
            self.figData.fig,
            label=''
        )


        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$g(i)$",
            # [0.5e1,0.5e3],[1,0.5e5],
            xScale='log',yScale='log',
            data=self.figData.fig
        )
        self.figData.SaveFig('BetweennessCentrality')

    def StrengthVsDegreeFig(self):
        si = self.si
        di = self.di
        dk = self.dk
        Nk = self.Nk

        figData = self.figData
        fig = figData.SetFigs()


        # Scatter
        sk = np.zeros_like(dk,dtype=float)
        for i,ki in enumerate(dk):
            v = np.nonzero(di == ki)[0]
            sk[i] = np.sum(si[v])
            sk[i] /= Nk[i]

        libF.CreateScatterPlot(dk,sk,figData.fig)

        # Fit in log–log space
        slope = libF.CreateLogRegressionPlot(dk,sk,figData.fig)

        sAvr = np.mean(si)
        libF.TextBlock(
            fig,[
                [libF.DataString(np.min(si),head=r's_{min}',space=False,formatVal='')],
                [libF.DataString(np.max(si),head=r's_{max}',space=False,formatVal='')],
                [libF.DataString(sAvr,head=r'\langle s\rangle',space=False,formatVal='.3f')],
                [libF.DataString(slope,head=r'\alpha',space=False,formatVal='.3f')]
            ],(0.75,0.25),
            (0,0.1)
        )


        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$s(k)$",
            xScale="log",yScale="log",
            data=figData.fig
        )
        figData.SaveFig('StrengthVsDegree')

    def AClusteringCoefficientFig(self):
        Nc = self.Nc
        A = self.A
        di = self.di
        dk = self.dk
        Nk = self.Nk

        figData = self.figData
        fig = figData.SetFigs()
        
        G = nx.from_numpy_array(A)
        C = nx.clustering(G)
        Cd = np.array([C[i] for i in range(Nc)])


        Ck = np.zeros_like(dk,dtype=float)
        for i,ki in enumerate(dk):
            v = di == ki
            Ck[i] = np.sum(Cd[v])/Nk[i]
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

        # CAvr = nx.average_clustering(G)
        CAvr = Ck.mean()
        libF.TextBlock(
            fig,[
                [libF.DataString(np.min(Cd),head=r'C_{min}',space=False,formatVal='.3f')],
                [libF.DataString(np.max(Cd),head=r'C_{max}',space=False,formatVal='.3f')],
                [libF.DataString(CAvr,head=r'\langle C\rangle',space=False,formatVal='.3f')]
            ],(0.75,0.75),
            (0,0.075)
        )

        libF.CreateScatterPlot(dk,Ck,figData.fig,label='')

        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$C(k)$",
            # [0,300],[0,0.8],
            data=figData.fig
        )
        figData.SaveFig('AClusteringCoefficient')

    def WClusteringCoefficientFig(self):
        A = self.A
        W = self.W

        si = self.si
        di = self.di
        dk = self.dk
        Nk = self.Nk
        Ck = self.Ck


        figData = self.figData
        fig, ax = figData.SetFigs(2)


        # Manual counting
        Cd = np.zeros_like(di,dtype=float)
        for i, ki in enumerate(di):
            if ki>1: # Consider only nodes with more than 2 neighbours
                vi = A[i,:]
                indices = np.nonzero(vi)[0]

                Ei = 0
                for n in indices:
                    for m in indices:
                        Ei += (W[i,n]+W[i,m])*A[n,m]
                Ei /= 2 # Divided by 2 since each edge is counted twice

                Cd[i] = Ei/(si[i]*(ki-1))
            else:
                Cd[i] = 0


        Ckw = np.zeros_like(dk,dtype=float)
        for i, ki in enumerate(dk):
            v = di == ki
            Ckw[i] = np.sum(Cd[v])/Nk[i]

        libF.CreateScatterPlot(dk,Ckw,figData.fig1,label='',ax=ax[0])

        # Style
        libF.SetFigStyle(
            r"$k$",r"$C^w(k)$",
            xScale="log",
            ax=ax[0],
            data=figData.fig1
        )


        CkwRel= (Ckw-Ck)/Ck
        libF.CreateScatterPlot(dk,CkwRel,figData.fig2,label='',ax=ax[1])
        self.CkwRel = CkwRel

        # Style
        libF.SetFigStyle(
            r"$k$",r"$C^w_{\text{rel}}(k)$",
            xScale="log",yScale="log",
            ax=ax[1],
            data=figData.fig2
        )


        # libF.CentreFig()
        figData.SaveFig('WClusteringCoefficient')

    def AAssortativityFig(self):
        A = self.A
        dk = self.dk

        figData = self.figData
        fig = figData.SetFigs()
        
        G = nx.from_numpy_array(A)
        ad = nx.average_neighbor_degree(G)
        ak = nx.average_degree_connectivity(G)

        aAvr = np.mean(list(ad.values()))
        libF.TextBlock(
            fig,[
                [libF.DataString(np.min(list(ad.values())),head=r'k_{nn}^{min}',space=False,formatVal='.3f')],
                [libF.DataString(np.max(list(ad.values())),head=r'k_{nn}^{max}',space=False,formatVal='.3f')],
                [libF.DataString(aAvr,head=r'\langle k_{nn}\rangle ',space=False,formatVal='.3f')]
            ],(0.75,0.75),
            (0,0.075)
        )

        knn = np.array([ak[ki] for ki in dk])
        libF.CreateScatterPlot(dk,knn,figData.fig,label='')
        self.knn = knn


        # Style
        # libF.CentreFig()
        libF.SetFigStyle(
            r"$k$",r"$k_{nn}(k)$",
            # [0,300],[40,95],
            data=figData.fig
        )
        figData.SaveFig('AAssortativity')

    def WAssortativityFig(self):
        W = self.W
        dk = self.dk
        knn = self.knn

        figData = self.figData
        fig, ax = figData.SetFigs(2)

        Gw = nx.from_numpy_array(W)
        # adw = nx.average_neighbor_degree(Gw,weight="weight")
        akw = nx.average_degree_connectivity(Gw,weight="weight")


        knnw = np.array([akw[ki] for ki in dk])
        libF.CreateScatterPlot(dk,knnw,figData.fig1,label='',ax=ax[0])

        # Style
        libF.SetFigStyle(
            r"$k$",r"$k_{nn}^w(k)$",
            xScale="log",yScale="log",
            ax=ax[0],
            data=figData.fig1
        )


        knnwRel = (knnw-knn)/knn
        libF.CreateScatterPlot(dk,knnwRel,figData.fig2,label='',ax=ax[1])

        # Style
        libF.SetFigStyle(
            r"$k$",r"$k_{nn,rel}^w(k)$",
            xScale="log",yScale="log",
            ax=ax[1],
            data=figData.fig2
        )


        # libF.CentreFig()
        figData.SaveFig('WAssortativity')

    def ShowFig(self):
        from matplotlib.pyplot import show
        show()
