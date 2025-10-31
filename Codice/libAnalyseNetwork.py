import os, sys
from pathlib import Path

import numpy as np
from scipy.stats import lognorm, linregress
import networkx as nx


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# import tikzplotlib as tikzpl # It's necessary «matplotlib<3.8»
# tikzpl.save(path)

import mplcursors
import mpld3

import subprocess



### Main code ###

def DegreeDistributionFig(di,Nn):

    fig = plt.figure()

    kAvr = np.sum(di)/Nn
    fig.text(
        0.8,0.25,
        r"$N$="+f"{Nn}"+"\n"+\
        r"$E$="+f"{int(np.sum(di)/2)}"+"\n"+\
        r"$k_{min}$="+f"{np.min(di)}"+"\n"+\
        r"$k_{max}$="+f"{np.max(di)}"+"\n"+\
        r"$\langle k\rangle $="+f"{kAvr}",
        ha="center",
        # fontsize=10,
        color="black"
    )


    # Histogram plot
    hgPlot = plt.hist(di,
        # bins=int(np.mean(d)),
        bins=25,
        # bins=60,
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label="Histogram"
    )

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    mplcursors.cursor(hgPlot[2],hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            "$P^{bin}(k)$="+f"{hgPlot[0][sel.index]:.3f}"
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
    x = np.linspace(0,np.max(di),500)

    shape, loc, scale = lognorm.fit(di,floc=0)
    plt.plot(
        [kAvr,kAvr],[0,lognorm.pdf(kAvr,shape,loc=loc,scale=scale)],
        label=r"Mean value $\langle k\rangle$",
        color="black",
        linewidth=1,
        linestyle="--"
    )

    fPlot = plt.plot(
        x,lognorm.pdf(x,shape,loc=loc,scale=scale),
        label="SciPy lognormal fit (ML)", # Maximum likelyhood
        color="blue",
        linewidth=1
    ) # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"k={sel.target[0]:.3f}, LN(k)={sel.target[1]:.3f}"
        )
    )

    # Style
    CentrePlot()
    SetPlotStyle(r"$k$",r"$P(k)$",[0,300],[0,0.03])
    SaveFig(fig,"DegreeDistributionSardegna")


def ClusteringCoefficientFig(A,di,dk,Nk):
    fig = plt.figure()
    
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
        r"$C_{min}$="f"{np.min(list(Cd.values())):.3f}\n"
        r"$C_{max}$="f"{np.max(list(Cd.values())):.3f}\n"
        r"$\langle C\rangle $="f"{CAvr:.3f}",
        ha="center",
        # fontsize=10,
        color="black"
    )

    sc = plt.scatter(dk,Ck,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f}, \
            cy={Ck[sel.index]:.3f}"
        )
    )

    # Style
    CentrePlot()
    SetPlotStyle(r"$k$",r"$C(k)$",[0,300],[0,0.8])
    SaveFig(fig,"ClusteringCoefficientSardegna")

    return Ck


def AssortativityFig(A,dk):
    fig = plt.figure()
    
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
    sc = plt.scatter(dk,knn,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]}, \
            cy={knn[sel.index]}"
        )
    )

    # Style
    CentrePlot()
    SetPlotStyle(r"$k$",r"$k_{nn}(k)$",[0,300],[40,95])
    SaveFig(fig,"AssortativitySardegna")

    return knn


def BetweennessCentralityFig(A,di):
    fig = plt.figure()
    
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
    sc = plt.scatter(di,bcd,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={di[sel.index]:.3f}, \
            cy={bcd[sel.index]:.3f}"
        )
    )

    # Style
    CentrePlot()
    SetPlotStyle(
        r"$k$",r"$g(i)$",
        # [0.5e1,0.5e3],[1,0.5e5],
        xScale="log",yScale="log"
    )
    SaveFig(fig,"BetweennessCentralitySardegna")


def WeightDistributionFig(wi):
    fig = plt.figure()

    # Histogram plot
    hgPlot = plt.hist(
        wi,
        bins=np.logspace(np.log10(min(wi)),np.log10(max(wi)),20),
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label="Histogram"
    )

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    mplcursors.cursor(hgPlot[2],hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            "$w^{bin}(k)$="+f"{hgPlot[0][sel.index]:.3f}"
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

    fPlot = plt.plot(
        binw,regression,
        label="Regression",
        color="blue",
        linewidth=1
    )

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
    CentrePlot()
    SetPlotStyle(
        r"$w$",r"$P(w)$",
        xScale="log",yScale="log"
    )
    SaveFig(fig,"WeightDistributionSardegna")


def StrengthDistributionFig(si):
    fig = plt.figure()

    # Histogram plot
    hgPlot = plt.hist(
        si,
        bins=np.logspace(
            np.log10(min(si)),
            np.log10(max(si)),
            20
        ),
        density=True,
        color="gray",
        edgecolor="none", # "black"
        label="Histogram"
    )

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

    fPlot = plt.plot(
        bins[v],regression,
        label="Regression",
        color="blue",
        linewidth=1
    )

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"k={sel.target[0]:.3f}, w(k)={sel.target[1]:.3f}"
        )
    )


    kAvr = np.mean(si)
    fig.text(
        0.75,0.5,
        r"$s_{min}$="f"{np.min(si)}\n"
        r"$s_{max}$="f"{np.max(si)}\n"
        r"$\langle s\rangle $="f"{kAvr:.3f}\n"
        r"$\alpha$="f"{slope:.3f}",
        ha="center",
        # fontsize=10,
        color="black"
    )


    # Style
    CentrePlot()
    SetPlotStyle(
        r"$s$",r"$P(s)$",
        xScale="log",yScale="log"
    )
    SaveFig(fig,"StrengthDistributionSardegna")


def StrengthFromDegreeFig(si,di,dk,Nk):
    fig = plt.figure()

    sk = np.zeros_like(dk,dtype=float)
    for i,ki in enumerate(dk):
        v = np.nonzero(di == ki)[0]
        sk[i] = np.sum(si[v])
        sk[i] /= Nk[i]

    sc = plt.scatter(
        dk,sk,
        label="Scatter plot",
        color='black',
        s=16
    )

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

    fPlot = plt.plot(
        dk,regression,
        label="Regression",
        color="blue",
        linewidth=1
    )

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
    CentrePlot()
    SetPlotStyle(
        r"$k$",r"$s(k)$",
        xScale="log",yScale="log"
    )
    SaveFig(fig,"StrengthFromDegreeSardegna")


def WeightedClusteringCoefficientFig(A,W,si,di,dk,Nk,Ck):
    fig, ax = plt.subplots(2,1)

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

    sc = ax[0].scatter(dk,Ckw,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f}, \
            cy={Ckw[sel.index]:.3f}"
        )
    )

    # Style
    SetPlotStyle(
        r"$k$",r"$C^w(k)$",
        xScale="log",ax=ax[0]
    )


    CkwRel= (Ckw-Ck)/Ck
    sc = ax[1].scatter(dk,CkwRel,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]:.3f}, \
            cy={CkwRel[sel.index]:.3f}"
        )
    )

    # Style
    SetPlotStyle(
        r"$k$",r"$C^w_{\text{rel}}(k)$",
        xScale="log",yScale="log",
        ax=ax[1]
    )


    CentrePlot()
    SaveFig(fig,"WeightedClusteringCoefficientSardegna")


def WeightedAssortativityFig(W,dk,knn):
    fig, ax = plt.subplots(2,1)

    Gw = nx.from_numpy_array(W)
    # adw = nx.average_neighbor_degree(Gw,weight="weight")
    akw = nx.average_degree_connectivity(Gw,weight="weight")


    knnw = np.array([akw[ki] for ki in dk])
    sc = ax[0].scatter(dk,knnw,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]}"
            f"cy={knn[sel.index]}"
        )
    )

    # Style
    SetPlotStyle(
        r"$k$",r"$k_{nn}^w(k)$",
        xScale="log",yScale="log",
        ax=ax[0]
    )


    knnwRel = (knnw-knn)/knn
    sc = ax[1].scatter(dk,knnwRel,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={dk[sel.index]}"
            f"cy={knnwRel[sel.index]}"
        )
    )

    # Style
    SetPlotStyle(
        r"$k$",r"$k_{nn,rel}^w(k)$",
        xScale="log",yScale="log",
        ax=ax[1]
    )


    CentrePlot()
    SaveFig(fig,"WeightedAssortativitySardegna")



### Auxiliary code ###

def CentrePlot():
    fig = plt.gcf()
    manager = plt.get_current_fig_manager()
    manager.window.update_idletasks()

    # Get screen size
    screenW = manager.window.winfo_screenwidth()
    screenH = manager.window.winfo_screenheight()

    # Save figure size in pixels
    figW, figH = fig.get_size_inches()*fig.dpi

    # Centre coordinates
    x = int((screenW-figW)/2)
    y = int((screenH-figH)/2)

    # Move the figure window
    manager.window.geometry(f"+{x}+{y}")


def SetTextStyle():
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 13
SetTextStyle()


def SetPlotStyle(
        xLabel=None,yLabel=None,
        xDom=None,yDom=None,
        xScale="linear",yScale="linear",
        xNotation="plain",yNotation="plain",
        ax=None
    ):
    if ax is None: ax = plt.gca()

    if xLabel: ax.set_xlabel(xLabel)
    if yLabel: ax.set_ylabel(yLabel)

    if xDom: ax.set_xlim(xDom)
    if yDom: ax.set_ylim(yDom)

    ax.set_xscale(xScale)
    ax.set_yscale(yScale)

    if xScale == "linear":
        plt.ticklabel_format(style=xNotation,axis="x",scilimits=(0,0))
    if yScale == "linear":
        plt.ticklabel_format(style=yNotation,axis="y",scilimits=(0,0))

    ax.grid(True,linestyle=":",linewidth=1)
    ax.set_axisbelow(True)

    _, labels = ax.get_legend_handles_labels()
    if any(labels): ax.legend()
    # Create a legend iff there are labels connected to graphs


def SaveFig(fig,name):
    pyFilePath = Path(__file__).resolve().parent.parent
    folderFig = pyFilePath/"Figure"
    Path(folderFig).mkdir(parents=True,exist_ok=True)

    def ext(ext): return (folderFig/name).with_suffix(ext)
    pdfFigPath  = ext(".pdf")
    pngFigPath  = ext(".png")
    htmlFigPath = ext(".html")

    
    plt.savefig(pdfFigPath,dpi=300,bbox_inches='tight')
    # plt.savefig(pngFigPath,dpi=300,bbox_inches='tight')
    # mpld3.save_html(fig,str(htmlFigPath))

    # Open the pdf file (cross-platform)
    if sys.platform.startswith("win"):
        os.startfile(pdfFigPath)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open",str(pdfFigPath)])
    else:
        subprocess.Popen(["xdg-open",str(pdfFigPath)])