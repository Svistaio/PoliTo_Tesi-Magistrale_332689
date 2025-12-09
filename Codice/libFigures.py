
import os, sys
from pathlib import Path
import subprocess

import numpy as np

from scipy.io import savemat
from scipy.stats import lognorm, linregress, pareto

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import mplcursors

global regCode


### Main functions ###

def CentreFig():
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

def SetFigStyle(
        xLabel=None,yLabel=None,
        xDom=None,yDom=None,
        xScale='lin',yScale='lin',
        xNotation='plain',yNotation='plain',
        ax=None,data=None
):
    if ax is None: ax = plt.gca()

    if xLabel: ax.set_xlabel(xLabel)
    if yLabel: ax.set_ylabel(yLabel)

    if xDom: ax.set_xlim(xDom)
    if yDom: ax.set_ylim(yDom)

    ax.set_xscale('linear' if xScale == 'lin' else xScale)
    ax.set_yscale('linear' if yScale == 'lin' else yScale)

    if xScale == 'lin':
        ax.ticklabel_format(style=xNotation,axis='x',scilimits=(0,0))
    if yScale == 'lin':
        ax.ticklabel_format(style=yNotation,axis='y',scilimits=(0,0))

    ax.grid(True,linestyle=":",linewidth=1)
    ax.set_axisbelow(True)

    _, labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend()
        data['style']['legend'] = True
    else:
        data['style']['legend'] = False
    # Create a legend iff there are labels connected to graphs

    data['style']['scale'] = {'x':xScale,'y':yScale}
    data['style']['labels'] = {'x':xLabel,'y':yLabel}

def SaveFig(
    name,folder,dicData
):
    global regCode

    projectPath = Path(__file__).resolve().parent.parent
    for format in ['.pdf','.mat','.tex']:
        folderPath = projectPath/'Figure'/format/regCode/folder
        Path(folderPath).mkdir(parents=True,exist_ok=True)
    # Define the folder path and create all the missing ones if necessary

    def figPath(figName,ext):
        return (projectPath/'Figure'/ext/regCode/folder/figName).with_suffix(ext)


    dicPath = {}

    format = '.pdf'
    dicPath[format] = figPath(name,format)
    savefig(dicPath[format],dpi=300,bbox_inches='tight')

    sumatraPath = r"..\.vscode\SumatraPDF.lnk"
    cmd = f'"{sumatraPath}" "{str(dicPath[format])}"'
    subprocess.Popen(cmd,shell=True)


    # def cmdTeX(matPath,texPath):
    #     m = matPath.as_posix()
    #     t = texPath.as_posix()

    #     cmd = (
    #         r'matlab -batch "mat2tex('
    #         fr"'{m}',"fr"'{t}'"r')"'
    #     )
    #     return cmd

    # for fig in dicData:
    #     if fig == 'fig':
    #         nome = name
    #     else:
    #         nome = name+fig

    #     format = '.mat'
    #     dicPath[format] = figPath(nome,format)
    #     savemat(dicPath[format],dicData[fig])

    #     format = '.tex'
    #     dicPath[format] = figPath(nome,format)

    #     cmd = cmdTeX(dicPath['.mat'],dicPath[format])
    #     subprocess.run(cmd,shell=True)

def CreateFunctionPlot(
    x,y,dicData,l=None,
    clr='black',idx='',ax=None
):
    if ax is None: ax = plt.gca()

    ax.plot(
        x,y,
        label=l,
        color=clr,
        linewidth=1,
    )

    dicData['plots'][f'functionPlot{idx}'] = {
        't':'function','x':x,'y':y,'l':l,'c':clr
    }

def CreateHistogramPlot(
    x,nBins,dicData,l='Histogram',
    clr='#808080',scale='lin',
    alfa=1,idx='',ax=None,norm=True
):
    if ax is None: ax = plt.gca()

    if scale == 'lin':
        hgPlot = ax.hist(
            x,
            bins=nBins, # 'auto'
            density=norm,
            color=clr,
            edgecolor="none", # "black"
            label=l,
            alpha=alfa
        )
    else:
        hgPlot = ax.hist(
            x,
            bins=np.logspace(
                np.log10(min(x)),
                np.log10(max(x)),
                nBins
            ),
            density=norm,
            color=clr,
            edgecolor="none", # "black"
            label=l,
            alpha=alfa
        )

    dicData['plots'][f'histogramPlot{idx}'] = {
        't':'histogram','x':x,'b':nBins,'l':l,'c':clr,'a':alfa,'s':scale
    }

    # hgPlot[0] = heights,
    # hgPlot[1] = bin edges,
    # hgPlot[2] = patches (Rectangle objects)

    mplcursors.cursor(hgPlot[2],hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"{hgPlot[0][sel.index]:.3f}"
        )
    )

    return hgPlot

def CreateScatterPlot(
    x,y,dicData,
    l='Scatter',clr='black',
    idx='',ax=None
):
    if ax is None: ax=plt.gca()

    sc = ax.scatter(
        x,y,
        label=l,
        color=clr,
        s=16
    )

    dicData['plots'][f'scatterPlot{idx}'] = {
        't':'scatter','x':x,'y':y,'l':l,'c':clr
    }

    mplcursors.cursor(sc,hover=True).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"({x[sel.index]},{y[sel.index]})"
        )
    )

def CreateLognormalFitPlot(
    v,dicData,lAvr='Average value',
    lFit='Lognormal fit (ML)', # Minimum likelihood
    clrAvr='black',clrFit='blue',
    idx='',ax=None
):
    if ax is None: ax=plt.gca()

    # Average value plot
    vAvr = np.mean(v)
    shape, loc, scale = lognorm.fit(v,floc=0)
    xM = [vAvr,vAvr]; yM = [0,lognorm.pdf(vAvr,shape,loc=loc,scale=scale)]
    ax.plot(
        xM,yM,label=lAvr,
        color=clrAvr,
        linewidth=1,
        linestyle="--"
    )

    # Fit plot
    xF = np.linspace(0,np.max(v),500)
    yF = lognorm.pdf(xF,shape,loc=loc,scale=scale)
    fPlot = ax.plot(
        xF,yF,label=lFit, # Maximum likelyhood
        color=clrFit,
        linewidth=1
    ) # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»

    dicData['plots'][f'meanPlot{idx}'] = {
        't':'function','x':xM,'y':yM,'l':lFit,'c':clrAvr
    }
    dicData['plots'][f'fitPlot{idx}'] = {
        't':'function','x':xF,'y':yF,'l':lAvr,'c':clrFit
    }

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"({sel.target[0]:.3f},{sel.target[1]:.3f})"
        )
    )

def CreateLogRegressionPlot(
    x,y,dicData,
    l='Regression',clr='blue',
    idx='',ax=None
):
    if ax is None: ax=plt.gca()

    logx  = np.log10(x)
    logy = np.log10(y)
    slope, intercept, _, _, _ = linregress(logx,logy)
    regression = 10**(intercept+slope*logx)

    fPlot = ax.plot(
        x,regression,label=l,
        color=clr,
        linewidth=1
    )
    
    dicData['plots'][f'regressionPlot{idx}'] = {
        't':'function','x':x,'y':regression,'l':l,'c':clr
    }

    mplcursors.cursor(fPlot,hover=False).connect(
        "add",lambda sel: sel.annotation.set_text(
            f"({sel.target[0]:.3f},{sel.target[1]:.3f})"
        )
    )

    return slope

def CreateParetoFitPlot(
    v,dicData,
    lSct='Scatter',lFit='Pareto fit (ML)', # Minimum likelihood
    clrSct='black',clrFit='black',
    idx='',ax=None
):
    if ax is None: ax=plt.gca()

    # Select the last quarter of city sizes
    vQuarter = np.percentile(v,75)
    vTail = v[v >= vQuarter]

    # Empirical CCDF on the tail
    vSort = np.sort(vTail) # Ascending values
    n = vSort.size
    ccdfEmp = 1 - np.arange(1,n+1,dtype=float)/n # P(X≥x)
    # Complementary Cumulative Distribution Function

    CreateScatterPlot(
        vSort,ccdfEmp,dicData,
        l=lSct,clr=clrSct,ax=ax
    )

    # Fitted CCDF from a Pareto function
    b, loc, scale = pareto.fit(vTail,floc=0,fscale=vQuarter) # b≈alpha
    ccdfFit = pareto.sf(vSort,b,loc=loc,scale=scale) # Survival function

    ax.plot(
        vSort,ccdfFit,label=lFit,
        color=clrFit
    )

    dicData['plots'][f'ScatterPlot{idx}'] = {
        't':'scatter','x':vSort,'y':ccdfEmp,'l':lSct,'c':clrSct
    }
    dicData['plots'][f'FitPlot{idx}'] = {
        't':'function','x':vSort,'y':ccdfFit,'l':lFit,'c':clrFit
    }

    return b

def CreateDicData(nFig):
    if not isinstance(nFig,int) or nFig <= 0:
        raise ValueError('The number of figures must be a integer positive number')

    dicData = {}
    if nFig == 1:
        dicData['fig'] = {'plots':{},'style':{}}
    else:
        for i in range(nFig):
            dicData[f'fig{i+1}'] = {'plots':{},'style':{}}
        
    return dicData


### Discarded code ###
#region From «SaveFig()» 
    #region Different naming rule between the folders
        # match folder:
        #     case 'NA':
        #         def ext(figName,ext): return (folderFig/figName).with_suffix(ext)
        #     case 'KS':
        #         def ext(figName,ext):
        #             figName = regCode['prefix']+figName
        #             return (folderFig/figName).with_suffix(ext)
    #endregion

    #region Open the pdf file (cross-platform)
        # if sys.platform.startswith("win"):
        #     os.startfile(pdfFigPath)
        # elif sys.platform.startswith("darwin"):
        #     subprocess.Popen(["open",str(pdfFigPath)])
        # else:
        #     subprocess.Popen(["xdg-open",str(pdfFigPath)])
    #endregion

    #region Other formats
        # pngFigPath  = ext(".png")
        # htmlFigPath = ext(".html")
        # plt.savefig(pngFigPath,dpi=300,bbox_inches='tight')
        # mpld3.save_html(fig,str(htmlFigPath))
    #endregion
#endregion
