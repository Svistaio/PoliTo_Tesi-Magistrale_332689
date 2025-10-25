import os, sys
from pathlib import Path

import numpy as np
from scipy.stats import lognorm
import networkx as nx


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 13

# import tikzplotlib as tikzpl # It's necessary «matplotlib<3.8»
# tikzpl.save(path)

import mplcursors
import mpld3

import subprocess

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


def SetFigStyle(
        xLabel=None,yLabel=None,
        xDom=None,yDom=None,
        xScale="linear",yScale="linear"
    ):
    if xLabel:
        plt.xlabel(xLabel)
    if yLabel:
        plt.ylabel(yLabel)

    if xDom:
        plt.xlim(xDom)
    if yDom:
        plt.ylim(yDom)

    plt.xscale(xScale)
    plt.yscale(yScale)

    plt.grid(True,linestyle=":",linewidth=1)
    plt.gca().set_axisbelow(True)
    _, labels = plt.gca().get_legend_handles_labels()
    if labels: # Create a legend iff there are labels connected to graphs
        plt.legend()

    CentrePlot()


def SaveFig(fig,name):
    pyFilePath = Path(__file__).resolve().parent.parent
    folderFig = pyFilePath/"Figure"
    Path(folderFig).mkdir(parents=True,exist_ok=True)

    def ext(ext): return (folderFig/name).with_suffix(ext)
    pdfFigPath  = ext(".pdf")
    pngFigPath  = ext(".png")
    htmlFigPath = ext(".html")

    
    plt.savefig(pdfFigPath,dpi=300,bbox_inches='tight')
    plt.savefig(pngFigPath,dpi=300,bbox_inches='tight')
    mpld3.save_html(fig,str(htmlFigPath))

    # Open the pdf file (cross-platform)
    if sys.platform.startswith("win"):
        os.startfile(pdfFigPath)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open",str(pdfFigPath)])
    else:
        subprocess.Popen(["xdg-open",str(pdfFigPath)])


def DegreeDistributionFig(d,N,k):

    fig = plt.figure()

    kAvr = np.sum(d)/len(d)
    fig.text(
        0.8,0.25,
        r"$N$="+f"{N}"+"\n"+\
        r"$E$="+f"{int(np.sum(d)/2)}"+"\n"+\
        r"$k_{min}$="+f"{min(d)}"+"\n"+\
        r"$k_{max}$="+f"{max(d)}"+"\n"+\
        r"$\langle k\rangle $="+f"{kAvr}",
        ha="center",
        # fontsize=10,
        color="black"
    )


    # Histogram plot
    hgPlot = plt.hist(d,
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
    x = np.linspace(0,max(k),500)

    shape, loc, scale = lognorm.fit(d,floc=0)
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
    SetFigStyle(r"$k$",r"$P(k)$",[0,300],[0,0.03])
    SaveFig(fig,"DegreeDistributionSardegna")


def ClusteringCoefficientFig(A,d,k,Nk):
    fig = plt.figure()
    
    G = nx.from_numpy_array(A)
    Cd = nx.clustering(G)

    Ck = np.zeros_like(k,dtype=float)
    for i,ki in enumerate(k):
        v = np.nonzero(d == ki)[0]
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
        r"$C_{min}$="+f"{min(list(Cd.values())):.3f}"+"\n"+\
        r"$C_{max}$="+f"{max(list(Cd.values())):.3f}"+"\n"+\
        r"$\langle C\rangle $="+f"{CAvr:.3f}",
        ha="center",
        # fontsize=10,
        color="black"
    )

    sc = plt.scatter(k,Ck,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={k[sel.index]:.3f}, \
            cy={Ck[sel.index]:.3f}"
        )
    )

    # Style
    SetFigStyle(r"$k$",r"$C(k)$",[0,300],[0,0.8])
    SaveFig(fig,"ClusteringCoefficientSardegna")


def AssortativityFig(A,k):
    fig = plt.figure()
    
    G = nx.from_numpy_array(A)
    ad = nx.average_neighbor_degree(G)
    ak = nx.average_degree_connectivity(G)

    aAvr = np.mean(list(ad.values()))
    fig.text(
        0.75,0.75,
        r"$k_{nn}^{min}$="+f"{min(list(ad.values())):.3f}"+"\n"+\
        r"$k_{nn}^{max}$="+f"{max(list(ad.values())):.3f}"+"\n"+\
        r"$\langle k_{nn}\rangle $="+f"{aAvr:.3f}",
        ha="center",
        # fontsize=10,
        color="black"
    )

    knn = np.array([ak[ki] for ki in k])
    sc = plt.scatter(k,knn,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={k[sel.index]}, \
            cy={knn[sel.index]}"
        )
    )

    # Style
    SetFigStyle(r"$k$",r"$k_{nn}(k)$",[0,300],[40,95])
    SaveFig(fig,"AssortativitySardegna")


def BetweennessCentralityFig(A,d):
    fig = plt.figure()
    
    G = nx.from_numpy_array(A)
    bc = nx.betweenness_centrality(G,normalized=False)

    aAvr = np.mean(list(bc.values()))
    fig.text(
        0.75,0.25,
        r"$g_{min}$="+f"{min(list(bc.values())):.3f}"+"\n"+\
        r"$g_{max}$="+f"{max(list(bc.values())):.0e}"+"\n"+\
        r"$\langle g\rangle $="+f"{aAvr:.3f}",
        ha="center",
        # fontsize=10,
        color="black"
    )

    bcd = np.array([bc[i] for i in range(len(d))])
    sc = plt.scatter(d,bcd,color='black',s=16)

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"cx={d[sel.index]:.3f}, \
            cy={bcd[sel.index]:.3f}"
        )
    )

    # Style
    SetFigStyle(
        r"$k$",r"$g(i)$",
        # [0.5e1,0.5e3],[1,0.5e5],
        xScale="log",yScale="log"
    )
    SaveFig(fig,"BetweennessCentralitySardegna")
