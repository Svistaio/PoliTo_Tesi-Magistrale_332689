import numpy as np
from scipy.stats import lognorm, linregress
import networkx as nx


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import mplcursors
import mpld3

import importlib

import libFigures as libF
importlib.reload(libF)


### Main functions ###

def DegreeDistributionFig(di,Nn):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}

    kAvr = np.sum(di)/Nn
    fig.text(
        0.8,0.25,
        fr"$N$={Nn}"f"\n"
        fr"$E$={int(np.sum(di)/2)}"f"\n"
        fr"$k_{{min}}$={np.min(di)}"f"\n"
        fr"$k_{{max}}$={np.max(di)}"f"\n"
        r"$\langle k\rangle $="f"{kAvr}",
        ha="center",
        # fontsize=10,
        color="black"
    )


    # Histogram plot
    x = di; nBins=25; l = "Histogram"
    hgPlot = plt.hist(x,
        # bins=int(np.mean(d)),
        bins=nBins,
        # bins=60,
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label=l
    )
    dicData['plots']['histogramPlot'] = {'t':'histogram','x':x,'b':nBins,'l':l}

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    mplcursors.cursor(hgPlot[2],hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            r"$P^{bin}(k)$="f"{hgPlot[0][sel.index]:.3f}"
        )
    )

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
    shape, loc, scale = lognorm.fit(di,floc=0)
    x = [kAvr,kAvr]; y = [0,lognorm.pdf(kAvr,shape,loc=loc,scale=scale)]
    l = r"Mean value $\langle k\rangle$"
    plt.plot(
        x,y,label=l,
        color="black",
        linewidth=1,
        linestyle="--"
    )
    dicData['plots']['meanPlot'] = {'t':'function','x':x,'y':y,'l':l}

    x = np.linspace(0,np.max(di),500)
    y = lognorm.pdf(x,shape,loc=loc,scale=scale)
    l = "SciPy lognormal fit (ML)"
    fPlot = plt.plot(
        x,y,
        label=l, # Maximum likelyhood
        color="blue",
        linewidth=1
    ) # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»
    dicData['plots']['fitPlot'] = {'t':'function','x':x,'y':y,'l':l}

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"k={sel.target[0]:.3f},LN(k)={sel.target[1]:.3f}"
        )
    )


    # Style
    libF.CentrePlot()

    xl = r"$k$"; yl = r"$P(k)$"
    libF.SetPlotStyle(xl,yl,[0,300],[0,0.03])
    dicData['style']['scale'] = {'x':'lin','y':'lin'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('DegreeDistribution','NA',dicData)

def AClusteringCoefficientFig(A,di,dk,Nk):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}
    
    G = nx.from_numpy_array(A)
    Cd = nx.clustering(G)

    Ck = np.zeros_like(dk,dtype=float)
    for i,ki in enumerate(dk):
        v = np.nonzero(di == ki)[0]
        for n in v:
            Ck[i] += Cd[n]
        Ck[i] /= Nk[i]

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

    x = dk; y = Ck; l = ''
    sc = plt.scatter(x,y,color='black',s=16)
    dicData['plots']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f},"
            f"cy={Ck[sel.index]:.3f}"
        )
    )

    # Style
    libF.CentrePlot()
    xl = r"$k$"; yl = r"$C(k)$"
    libF.SetPlotStyle(xl,yl,[0,300],[0,0.8])
    dicData['style']['scale'] = {'x':'lin','y':'lin'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('AClusteringCoefficient','NA',dicData)

    return Ck

def AAssortativityFig(A,dk):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}
    
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
    x = dk; y = knn; l = ''
    sc = plt.scatter(x,y,color='black',s=16)
    dicData['plots']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]}, \
            cy={knn[sel.index]}"
        )
    )

    # Style
    libF.CentrePlot()

    xl = r"$k$"; yl = r"$k_{nn}(k)$"
    libF.SetPlotStyle(xl,yl,[0,300],[40,95])
    dicData['style']['scale'] = {'x':'lin','y':'lin'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('AAssortativity','NA',dicData)

    return knn

def BetweennessCentralityFig(A,di):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}
    
    G = nx.from_numpy_array(A)
    bc = nx.betweenness_centrality(G,normalized=False)

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

    bcd = np.array([bc[i] for i in range(len(di))])
    x = di; y = bcd; l = ''
    sc = plt.scatter(x,y,color='black',s=16)
    dicData['plots']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={di[sel.index]:.3f}, \
            cy={bcd[sel.index]:.3f}"
        )
    )

    # Style
    libF.CentrePlot()

    xl = r"$k$"; yl = r"$g(i)$"
    libF.SetPlotStyle(
        xl,yl,
        # [0.5e1,0.5e3],[1,0.5e5],
        xScale="log",yScale="log"
    )
    dicData['style']['scale'] = {'x':'log','y':'log'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('BetweennessCentrality','NA',dicData)

def WeightDistributionFig(wi):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}


    # Histogram plot
    x = wi; nBins = 20
    l = "Histogram"
    hgPlot = plt.hist(
        x,
        bins=np.logspace(np.log10(min(wi)),np.log10(max(wi)),20),
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label=l
    )
    dicData['plots']['histogramPlot'] = {'t':'histogram','x':x,'b':nBins,'l':l}

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    mplcursors.cursor(hgPlot[2],hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            "$w^{bin}(k)$="f"{hgPlot[0][sel.index]:.3f}"
        )
    )


    # SciPy regression
    binw = (hgPlot[1][1:]+hgPlot[1][:-1])/2
    binPw = hgPlot[0]
    # sc = plt.scatter(binw,binPw,c='r',s=50)

    # Fit in log–log space
    logBinw  = np.log10(binw)
    logBinPw = np.log10(binPw)
    slope, intercept, _, _, _ = linregress(logBinw,logBinPw)
    regression = 10**(intercept+slope*logBinw)

    x = binw; y = regression; l = "Regression"
    fPlot = plt.plot(
        x,y,label=l,
        color="blue",
        linewidth=1
    )
    dicData['plots']['regressionPlot'] = {'t':'function','x':x,'y':y,'l':l}

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"k={sel.target[0]:.3f}, w(k)={sel.target[1]:.3f}"
        )
    )


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
    libF.CentrePlot()

    xl = r"$w$"; yl = r"$P(w)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",yScale="log"
    )
    dicData['style']['scale'] = {'x':'log','y':'log'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('WeightDistribution','NA',dicData)

def StrengthDistributionFig(si):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}


    # Histogram plot
    x = si; nBins=20; l = "Histogram"
    hgPlot = plt.hist(
        x,
        bins=np.logspace(
            np.log10(min(x)),
            np.log10(max(x)),
            nBins
        ),
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label=l
    )
    dicData['plots']['histogramPlot'] = {'t':'histogram','x':x,'b':nBins,'l':l}

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    mplcursors.cursor(hgPlot[2],hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            "$w^{bin}(k)$="+f"{hgPlot[0][sel.index]:.3f}"
        )
    )


    # SciPy regression
    bins = (hgPlot[1][1:]+hgPlot[1][:-1])/2
    binPs = hgPlot[0]
    # sc = plt.scatter(binw,binPw,c='r',s=50)

    # Fit in log–log space
    v = binPs>0; v[:6] = 0
    logBins  = np.log10(bins[v])
    logBinPs = np.log10(binPs[v])
    slope, intercept, _, _, _ = linregress(logBins,logBinPs)
    regression = 10**(intercept+slope*logBins)

    x = bins[v]; y = regression; l = "Regression"
    fPlot = plt.plot(
        x,y,label=l,
        color="blue",
        linewidth=1
    )
    dicData['plots']['fitPlot'] = {'t':'function','x':x,'y':y,'l':l}

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"k={sel.target[0]:.3f}, w(k)={sel.target[1]:.3f}"
        )
    )


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
    libF.CentrePlot()

    xl = r"$s$"; yl = r"$P(s)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",yScale="log"
    )
    dicData['style']['scale'] = {'x':'log','y':'log'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('StrengthDistribution','NA',dicData)

def StrengthFromDegreeFig(si,di,dk,Nk):
    fig = plt.figure()
    dicData = {'plots':{},'style':{}}


    # Scatter
    sk = np.zeros_like(dk,dtype=float)
    for i,ki in enumerate(dk):
        v = np.nonzero(di == ki)[0]
        sk[i] = np.sum(si[v])
        sk[i] /= Nk[i]

    x = dk; y = sk; l = "Scatter plot"
    sc = plt.scatter(
        x,y,label=l,
        color='black',
        s=16
    )
    dicData['plots']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f}, \
            cy={sk[sel.index]:.3f}"
        )
    )


    # Fit in log–log space
    logdk = np.log10(dk); logsk = np.log10(sk)
    slope, intercept, _, _, _ = linregress(logdk,logsk)
    regression = 10**(intercept+slope*logdk)

    x = dk; y = regression; l = "Regression"
    fPlot = plt.plot(
        x,y,label=l,
        color="blue",
        linewidth=1
    )
    dicData['plots']['regressionPlot'] = {'t':'function','x':x,'y':y,'l':l}

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"k={sel.target[0]:.3f}, w(k)={sel.target[1]:.3f}"
        )
    )

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
    libF.CentrePlot()

    xl = r"$k$"; yl = r"$s(k)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",yScale="log"
    )
    dicData['style']['scale'] = {'x':'log','y':'log'}
    dicData['style']['labels'] = {'x':xl,'y':yl}

    libF.SaveFig('StrengthFromDegree','NA',dicData)

def WClusteringCoefficientFig(A,W,si,di,dk,Nk,Ck):
    fig, ax = plt.subplots(2,1)
    dicData = {'plots':{'fig1':{},'fig2':{}},'style':{'fig1':{},'fig2':{}}}


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

    x = dk; y = Ckw; l =''
    sc = ax[0].scatter(x,y,color='black',s=16)
    dicData['plots']['fig1']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f}, \
            cy={Ckw[sel.index]:.3f}"
        )
    )

    # Style
    xl = r"$k$"; yl = r"$C^w(k)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",ax=ax[0]
    )
    dicData['style']['fig1']['scale'] = {'x':'log','y':'lin'}
    dicData['style']['fig1']['labels'] = {'x':xl,'y':yl}


    CkwRel= (Ckw-Ck)/Ck
    x = dk; y = CkwRel; l = ''
    sc = ax[1].scatter(x,y,color='black',s=16)
    dicData['plots']['fig2']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f}, \
            cy={CkwRel[sel.index]:.3f}"
        )
    )

    # Style
    xl = r"$k$"; yl = r"$C^w_{\text{rel}}(k)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",yScale="log",
        ax=ax[1]
    )
    dicData['style']['fig2']['scale'] = {'x':'log','y':'log'}
    dicData['style']['fig2']['labels'] = {'x':xl,'y':yl}


    libF.CentrePlot()
    libF.SaveFig('WClusteringCoefficient','NA',dicData,figs=True)

def WAssortativityFig(W,dk,knn):
    fig, ax = plt.subplots(2,1)
    dicData = {'plots':{'fig1':{},'fig2':{}},'style':{'fig1':{},'fig2':{}}}

    Gw = nx.from_numpy_array(W)
    # adw = nx.average_neighbor_degree(Gw,weight="weight")
    akw = nx.average_degree_connectivity(Gw,weight="weight")


    knnw = np.array([akw[ki] for ki in dk])
    x = dk; y = knnw; l = ''
    sc = ax[0].scatter(x,y,color='black',s=16)
    dicData['plots']['fig1']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]}"
            f"cy={knn[sel.index]}"
        )
    )

    # Style
    xl = r"$k$"; yl = r"$k_{nn}^w(k)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",yScale="log",
        ax=ax[0]
    )
    dicData['style']['fig1']['scale'] = {'x':'log','y':'log'}
    dicData['style']['fig1']['labels'] = {'x':xl,'y':yl}


    knnwRel = (knnw-knn)/knn
    x = dk; y = knnwRel; l = ''
    sc = ax[1].scatter(x,y,color='black',s=16)
    dicData['plots']['fig2']['scatterPlot'] = {'t':'scatter','x':x,'y':y,'l':l}

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]}"
            f"cy={knnwRel[sel.index]}"
        )
    )

    # Style
    xl = r"$k$"; yl = r"$k_{nn,rel}^w(k)$"
    libF.SetPlotStyle(
        xl,yl,
        xScale="log",yScale="log",
        ax=ax[1]
    )
    dicData['style']['fig2']['scale'] = {'x':'log','y':'log'}
    dicData['style']['fig2']['labels'] = {'x':xl,'y':yl}


    libF.CentrePlot()
    libF.SaveFig('WAssortativity','NA',dicData,figs=True)
