
# import os, sys
from pathlib import Path
import subprocess

import numpy as np

from scipy.io import savemat
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
# import mplcursors


### Main Functions and class ###

# Style and data
def SetTextStyle():
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 13
SetTextStyle()

def SetFigStyle(
    xLabel=None,
    yLabel=None,
    xDom=None,
    yDom=None,
    xScale='lin',
    yScale='lin',
    xNotation='plain',
    yNotation='plain',
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

class FigData():
    def __init__(self,clsPrm,folder):
        self.folder = folder
        self.projectPath = Path(__file__).resolve().parent.parent

        self.regCode = str(clsPrm.region)
        self.PdfPopUp = clsPrm.PdfPopUp
        self.LaTeXConversion = clsPrm.LaTeXConversion

    def SetFigs(self,nFig):
        if not isinstance(nFig,int) or nFig <= 0:
            raise ValueError('The number of figures must be a integer positive number')

        if nFig == 1:
            self.fig = {'plots':{},'style':{}}
        else:
            for i in range(nFig):
                setattr(self,f'fig{i+1}',{'plots':{},'style':{}})

        self.nFig = nFig

    def SaveFig(self,name):
        self.name = [name]
        self.SaveFig2pdf()

        if self.nFig>1:
            self.name = [f'{self.name}fig{f+1}' for f in range(self.nFig)]

        if self.LaTeXConversion:
            for f in range(self.nFig):
                self.SaveFig2mat(f)
                self.SaveFig2tex(f)

    def SaveFig2pdf(self):
        format = '.pdf'
        self.CreateFolders(format)
        pdfPath = self.FigPath(self.name[0],format)

        savefig(
            pdfPath,
            dpi=300,
            bbox_inches='tight'
        )

        if self.PdfPopUp:
            sumatraPath = r"..\.vscode\SumatraPDF.lnk"
            cmd = f'"{sumatraPath}" "{str(pdfPath)}"'
            subprocess.Popen(cmd,shell=True)

    def SaveFig2mat(self,f):
        format = '.mat'
        self.CreateFolders(format)

        self.matPath = self.FigPath(self.name[f],format)
        savemat(self.matPath,getattr(self,f'fig{f+1}'))

    def SaveFig2tex(self,f):
        format = '.tex'
        self.CreateFolders(format)

        texPath = self.FigPath(self.name[f],format)
        cmd = self.cmdTeX(self.matPath,texPath)
        subprocess.run(cmd,shell=True)

    def CreateFolders(self,format):
        folderPath = self.projectPath/'Figure'/format/self.regCode/self.folder
        Path(folderPath).mkdir(parents=True,exist_ok=True)
        # Define the folder path and create all the missing ones if necessary

    def FigPath(self,figName,ext):
        return (
            self.projectPath/'Figure'/
            ext/self.regCode/
            self.folder/figName
        ).with_suffix(ext)

    def cmdTeX(matPath,texPath):
        m = matPath.as_posix()
        t = texPath.as_posix()

        cmd = (
            r'matlab -batch "mat2tex('
            fr"'{m}',"fr"'{t}'"r')"'
        )
        return cmd

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

# Primary/Basic plots
def CreateFunctionPlot(
    x,y,
    figData,
    Ni=1,
    ta=None,
    yErr=True,
    label=None,
    linestyle='-',
    linewidth=1,
    color='black',
    idx='',
    ax=None
):
    if ax is None: ax = plt.gca()

    if Ni == 1:
        if y.ndim != 1: y = y.ravel()
        ax.plot(
            x,y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )
        figData['plots'][f'functionPlot{idx}'] = {
            't':'function','x':x,'y':y,'l':label,'c':color
        }

    else:
        if yErr:
            avrFunc, err95Func = EvaluateConfidenceInterval(y,ta,Ni)
            y = avrFunc; xerr95 = None; yerr95 = err95Func
        else:
            avrFunc, err95Func = EvaluateConfidenceInterval(x,ta,Ni)
            x = avrFunc; xerr95 = err95Func; yerr95 = None

        ax.plot(
            x,y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )
        figData['plots'][f'functionPlot{idx}'] = {
            't':'function','x':x,'y':y,'l':label,'c':color
        }

        CreateConfidenceIntervalPlot(
            x,y,
            figData,
            typ='functionfill',
            xerr95=xerr95,
            yerr95=yerr95,
            alfa=0.5,
            clr=color,
            ax=ax,
            idx=idx
        )

    # mplcursors.cursor(fPlot,hover=False).connect(
    #     "add",lambda sel: sel.annotation.set_text(
    #         f"({sel.target[0]:.3f},{sel.target[1]:.3f})"
    #     )
    # )

def CreateScatterPlot(
    x,y,
    figData,
    size=16,
    Ni=1,
    ta=None,
    yErr=True,
    label='Scatter',
    color='black',
    idx='',
    ax=None,
    alpha=1
):
    if ax is None: ax=plt.gca()

    if Ni == 1:
        if y.ndim != 1: y = y.ravel()
        sct = ax.scatter(
            x,y,
            label=label,
            color=color,
            s=size,
            alpha=alpha
        )
        figData['plots'][f'scatterPlot{idx}'] = {
            't':'scatter','x':x,'y':y,'l':label,'c':color
        }
    else:
        if yErr:
            avrSct, err95Sct = EvaluateConfidenceInterval(y,ta,Ni)

            sct = ax.scatter(
                x,
                avrSct,
                label=label,
                color=color,
                s=size,
                alpha=alpha
            )
            figData['plots'][f'functionPlot{idx}'] = {
                't':'function','x':x,'y':avrSct,'l':label,'c':color
            }

            CreateConfidenceIntervalPlot(
                x,
                avrSct,
                figData,
                typ='errorbar',
                yerr95=err95Sct,
                clr=color,
                ax=ax,
                idx=idx,
                alfa=0.5
            )

        else:
            avrSct, err95Sct = EvaluateConfidenceInterval(x,ta,Ni)

            sct = ax.scatter(
                avrSct,
                y,
                label=label,
                color=color,
                s=size,
                alpha=alpha
            )
            figData['plots'][f'functionPlot{idx}'] = {
                't':'function','x':avrSct,'y':y,'l':label,'c':color
            }

            CreateConfidenceIntervalPlot(
                avrSct,
                y,
                figData,
                typ='errorbar',
                xerr95=err95Sct,
                clr=color,
                ax=ax,
                idx=idx,
                alfa=0.5
            )

    # mplcursors.cursor(sct,hover=True).connect(
    #     "add",lambda sel: sel.annotation.set_text(
    #         f"({x[sel.index]},{y[sel.index]})"
    #     )
    # )

def CreateHistogramPlot(
    x,nBins,
    figData,
    limits=None,
    scale='lin',
    Ni=1,
    ta=None,
    label='Histogram',
    color='#808080',
    norm=True,
    alpha=1,
    idx='',
    ax=None
):
    if ax is None: ax = plt.gca()

    def HistogramPlot(
        binEdges,
        binMidPoints,
        binAverages
    ):
        hgPlot = ax.hist(
            binMidPoints,
            bins=binEdges, # 'auto'
            weights=binAverages,
            # density=norm,
            color=color,
            edgecolor="none", # "black"
            label=label,
            alpha=alpha
        )

        figData['plots'][f'histogramPlot{idx}'] = {
            't':'histogram','x':binAverages,'b':binEdges,
            'l':label,'c':color,'a':alpha,'s':'log'
        }

        # hgPlot[0] = heights,
        # hgPlot[1] = bin edges,
        # hgPlot[2] = patches (Rectangle objects)

        # mplcursors.cursor(hgPlot[2],hover=False).connect(
        #     "add",lambda sel: sel.annotation.set_text(
        #         f"{hgPlot[0][sel.index]:.3f}"
        #     )
        # )

    if limits is None:
        xMin = np.min(x); xMax=np.max(x)
    else:
        xMin = limits[0]; xMax=limits[1]

    if scale == 'lin':
        binPoints = np.linspace(
            xMin,
            xMax,
            num=2*nBins+1
        )
    else:
        binPoints = np.logspace(
            np.log10(xMin),
            np.log10(xMax),
            num=2*nBins+1
        )

    binEdges = binPoints[0::2]
    binMidPoints = binPoints[1::2]
    # binMidPoints = (binEdges[:-1]+binEdges[1:])/2 # It only works in the linear case

    if Ni == 1:
        if x.ndim != 1: x = x.ravel()

        binAverages, _ = np.histogram(x,binEdges,density=norm)
        HistogramPlot(binEdges,binMidPoints,binAverages)

        return binMidPoints, binAverages
    else:
        hgData = [None]*Ni
        for r in range(Ni):
            hgData[r] = np.histogram(x[r,:],binEdges,density=norm)

        binAverages, hgErr95 = EvaluateConfidenceInterval(
            [hgData[r][0] for r in range(Ni)],ta,Ni
        )

        HistogramPlot(binEdges,binMidPoints,binAverages)

        # To avoid [possible] lower negative confidence intervals in the lower and upper tails, they have to be clipped
        lowerBinEstimate = np.maximum(binAverages-hgErr95,0)
        upperBinEstimate = binAverages+hgErr95
        binErr95 = np.array([
            binAverages-lowerBinEstimate,
            upperBinEstimate-binAverages
        ])

        CreateConfidenceIntervalPlot(
            binMidPoints,
            binAverages,
            figData,
            typ='errorbar',
            yerr95=binErr95,
            clr=color,
            ax=ax,
            idx=idx,
            alfa=alpha
        )

        return binMidPoints, binAverages

# Secondary/Compound plots
def CreateLogRegressionPlot(
    x,y,
    figData,
    l='Regression',
    color='blue',
    idx='',
    ax=None
):
    if ax is None: ax=plt.gca()

    xp = x[y>0]; yp = y[y>0]
    logx  = np.log10(xp); logy = np.log10(yp)

    slope, intercept, _, _, _ = stats.linregress(logx,logy)
    regression = 10**(intercept+slope*logx)

    fPlot = ax.plot(
        xp,regression,
        label=l,
        color=color,
        linewidth=1
    )
    
    figData['plots'][f'regressionPlot{idx}'] = {
        't':'function','x':x,'y':regression,'l':l,'c':color
    }

    # mplcursors.cursor(fPlot,hover=False).connect(
    #     "add",lambda sel: sel.annotation.set_text(
    #         f"({sel.target[0]:.3f},{sel.target[1]:.3f})"
    #     )
    # )

    return slope

def CreateLognormalFitPlot(
    v,
    figData,
    Ni=1,
    ta=None,
    labelAvr='Average value',
    labelFit='Lognormal fit (ML)', # Minimum likelihood
    colorAvr='black',
    colorFit='blue',
    idx='',
    ax=None
):
    if ax is None: ax=plt.gca()

    xF = np.logspace(
        np.log10(np.quantile(v,0)),
        np.log10(np.quantile(v,1)),
        500
    )

    if Ni == 1:
        shape, loc, scale = stats.lognorm.fit(v,floc=0)
        # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»
        yF = stats.lognorm.pdf(xF,shape,loc=loc,scale=scale)

        vAvr = np.mean(v)
        # xM = np.array([vAvr,vAvr])
        # yM = np.array([0,stats.lognorm.pdf(vAvr,shape,loc=loc,scale=scale)])
    else:
        fitData = [None]*Ni
        avrData = [None]*Ni
        yM = 0

        for r in range(Ni):
            shape, loc, scale = stats.lognorm.fit(v[r,:],floc=0)
            fitData[r] = stats.lognorm.pdf(xF,shape,loc=loc,scale=scale)

            avrData[r] = np.mean(v[r,:])
            yM += stats.lognorm.pdf(avrData[r],shape,loc=loc,scale=scale)/Ni

        yF = fitData

        # xM = np.transpose(np.array([np.array(avrData)]*2))
        # yM = np.array([0,yM])

    # Fit plot
    CreateFunctionPlot(
        xF,yF,
        figData,
        Ni=Ni,
        ta=ta,
        label=labelFit,
        # linewidth=1,
        color=colorFit,
        idx=idx,
        ax=ax
    )

    # Average value plot
    # CreateFunctionPlot(
    #     xM,yM,
    #     figData,
    #     Ni=Ni,
    #     ta=ta,
    #     yErr=False,
    #     label=labelAvr,
    #     # linewidth=1,
    #     linestyle="--",
    #     color=colorAvr,
    #     idx=idx,
    #     ax=ax
    # )

def CreateParetoFitPlot(
    v,
    figData,
    Ni=1,
    ta=None,
    labelSct='Scatter',
    labelFit='Pareto fit (ML)', # Minimum likelihood
    colorSct='black',
    colorFit='black',
    idx='',
    ax=None
):
    if ax is None: ax=plt.gca()

    xF = np.logspace(
        np.log10(np.quantile(v,0.75)),
        np.log10(np.quantile(v,1)),
        100
    )

    if Ni == 1:
        # Select the last quarter of city sizes
        vQuarter = np.quantile(v,.75)
        vTail = v[v >= vQuarter]

        # Empirical CCDF on the tail
        vSort = np.sort(vTail) # Ascending values
        n = vSort.size
        ccdfEmp = 1-np.arange(1,n+1,dtype=float)/n # P(X≥x)
        # Complementary Cumulative Distribution Function
        # Formally it should be «1-np.arange(1,n+1,dtype=float)/Nc-(Nc-n)/Nc», but since the only relevant information is the Pareto index, the exact probability value can be omitted
        xS = vSort; yS = ccdfEmp

        b, loc, scale = stats.pareto.fit(vTail,floc=0,fscale=vQuarter) # b≈alpha
        ccdfFit = stats.pareto.sf(xF,b,loc=loc,scale=scale) # Survival function
        yF = ccdfFit

    else:
        vTails  = [None]*Ni
        b       = [None]*Ni
        ccdfFit = [None]*Ni

        for r in range(Ni):
            # Select the last quarter of city sizes
            vQuarter = np.quantile(v[r,:],.75)
            vTail = v[r,v[r,:] >= vQuarter]

            # Empirical CCDF on the tail
            vTails[r] = np.sort(vTail) # Ascending values

            # Fitted CCDF from a Pareto function
            b[r], loc, scale = stats.pareto.fit(vTail,floc=0,fscale=vQuarter) # b≈alpha
            ccdfFit[r] = stats.pareto.sf(xF,b[r],loc=loc,scale=scale) # Survival function

        yF = ccdfFit

        n = vTails[0].size
        ccdfEmp = 1-np.arange(1,n+1,dtype=float)/n # P(X≥x)
        # Complementary Cumulative Distribution Function
        # Formally it should be «1-np.arange(1,n+1,dtype=float)/Nc-(Nc-n)/Nc», but since the only relevant information is the Pareto index, the exact probability value can be omitted
        xS = vTails; yS = ccdfEmp

    CreateScatterPlot(
        xS,yS,
        figData,
        Ni=Ni,
        ta=ta,
        size=10,
        yErr=False,
        label=labelSct,
        color=colorSct,
        ax=ax
    )

    CreateFunctionPlot(
        xF,yF,
        figData,
        Ni=Ni,
        ta=ta,
        label=labelFit,
        # linewidth=1,
        color=colorFit,
        ax=ax,
        idx=idx
    )

    return b


### Auxiliary functions ###

def EvaluateConfidenceInterval(
    data,
    ta,
    Ni
):
    if Ni>1:
        values = np.array(data)
        averages = np.mean(values,axis=0)
        err95 = ta*np.std(values,axis=0,ddof=1)/np.sqrt(Ni)
        return (averages,err95)
    else:
        if isinstance(data,np.ndarray):
            return (data[0],None)
        else:
            return (data,None)

def CreateConfidenceIntervalPlot(
    x,
    y,
    figData,
    typ='errorbar',
    yerr95=None,
    xerr95=None,
    clr='black',
    fmt='none',
    ax=None,
    idx='',
    alfa=1
):
    if ax is None: ax = plt.gca()

    match typ:
        case 'errorbar':
            ax.errorbar(
                x,y,
                xerr=xerr95,
                yerr=yerr95,
                ecolor=clr,
                fmt=fmt,
                linestyle='none',
                elinewidth=1.2,
                capsize=8,
                capthick=1.6,
                markersize=0,
                markerfacecolor=clr,
                markerfacecoloralt=clr,
                markeredgecolor=clr,
                markeredgewidth=0,
                alpha=alfa
            )
            figData['plots'][f'{typ}{idx}'] = {
                't':typ,'x':x,'y':y,'ye':yerr95,
                'l':'','c':clr,'a':alfa,
            }

        case 'functionfill':
            if xerr95 is None:
                ax.fill_between(
                    x,y+yerr95,y-yerr95,
                    facecolor=clr,
                    alpha=alfa,
                    linewidth=0
                )
                figData['plots'][f'{typ}y{idx}'] = {
                    't':typ,'x':x,'y':y,'ye':yerr95,
                    'l':'','c':clr,'a':alfa,
                }
            else:
                ax.fill_betweenx(
                    y,x+xerr95,x-xerr95,
                    facecolor=clr,
                    alpha=alfa,
                    linewidth=0
                )
                figData['plots'][f'{typ}x{idx}'] = {
                    't':typ,'x':x,'y':y,'xe':xerr95,
                    'l':'','c':clr,'a':alfa,
                }


### Discarded code ###

#region From «FigData.SaveFig()» 
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
