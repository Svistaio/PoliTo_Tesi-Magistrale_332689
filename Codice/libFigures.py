
import os, sys
from pathlib import Path

import subprocess

import matplotlib.pyplot as plt


### Main functions ###

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
        ax.ticklabel_format(style=xNotation,axis="x",scilimits=(0,0))
    if yScale == "linear":
        ax.ticklabel_format(style=yNotation,axis="y",scilimits=(0,0))

    ax.grid(True,linestyle=":",linewidth=1)
    ax.set_axisbelow(True)

    _, labels = ax.get_legend_handles_labels()
    if any(labels): ax.legend()
    # Create a legend iff there are labels connected to graphs

def SaveFig(fig,name):
    # Add a cycle for each format (.pdf, .mat and .tex)

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
