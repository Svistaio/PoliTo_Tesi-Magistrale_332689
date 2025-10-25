import os, sys
from pathlib import Path

from zipfile import ZipFile as ZF
import io

import numpy as np
from scipy.stats import lognorm

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

### Functions ###

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


def SetFigStyle(xLabel,yLabel,xDom,yDom):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.xlim(xDom)
    plt.ylim(yDom)

    plt.grid(True,linestyle=":",linewidth=1)
    plt.legend()

    CentrePlot()


def SaveFig(name,fig):
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

    fig.text(
        0.8,0.25,
        "$N$="+f"{N}"+"\n"+\
        "$E$="+f"{int(np.sum(d)/2)}"+"\n"+\
        "$k_{min}$="+f"{min(d)}"+"\n"+\
        "$k_{max}$="+f"{max(d)}"+"\n"+\
        r"$\langle k\rangle $="+f"{np.sum(d)/len(d)}",
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
    SaveFig("DegreeDistributionSardegna",fig)

    # plt.show()
    # mpld3.show()


### Data ###
A = np.array([],dtype=int)

with ZF("../Dati/AdjacencyMatricesIt91.zip") as z:
    with z.open("20AdjacencyMatrixSardegna.txt","r") as f:
        A = np.loadtxt( # Decodes binary stream as UTF-8 text
            io.TextIOWrapper(f,encoding="utf-8"),
            delimiter=",",
            dtype=int
        )

# Vectors of degrees
d = np.sum(A, axis=0)
N = len(d) # Number of nodes (normalization factor)

# Unique degrees and corresponding frequencies
k, counts = np.unique(d,return_counts=True)
# Pk = counts/N

DegreeDistributionFig(d,N,k)