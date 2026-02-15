
# Library to create figures

# import os, sys
from pathlib import Path
import subprocess

import numpy as np

from scipy.io import savemat
from scipy import stats
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
# import mplcursors

from libParameters import projectFolder


### Main class and functions ###

# Figure data
class FigData():
    def __init__(self,clsPrm,folder):
        self.folder = folder
        self.projectFolder = projectFolder

        self.regCode = str(clsPrm.region)
        self.PdfPopUp = clsPrm.PdfPopUp
        self.LaTeXConversion = clsPrm.LaTeXConversion

        for ext in ('.pdf','.mat','.tex'):
            if ext=='.pdf' or self.LaTeXConversion:
                setattr(self,f'{ext[1:]}FolderPath',self.CreateFolders(ext))
                # for p in folder.iterdir():
                #     if p.is_file():
                #         p.unlink()

    def SetFigs(self,nRow=1,nCol=1,size=None):
        nFig = nRow*nCol; self.nFig = nFig

        if not isinstance(nFig,int) or nFig <= 0:
            raise ValueError('The number of figures must be a integer positive number')

        for i in range(nFig):
            setattr(self,f'fig{i+1}',{'plots':{},'style':{}})

        if nFig == 1:
            self.fig = self.fig1
            return plt.figure()
        else:
            return plt.subplots(
                nRow,nCol,
                figsize=size,
                gridspec_kw=dict(
                    wspace=0.2,
                    hspace=0.2
                )
            )

    def SaveFig(self,name):
        self.SaveFig2pdf(name)

        self.names = [
            f'{name}fig{f+1}' for f in range(self.nFig)
        ] if self.nFig>1 else [name]

        if self.LaTeXConversion:
            for f in range(self.nFig):
                self.SaveFig2mat(f)
                self.SaveFig2tex(f)

    def SaveFig2pdf(self,name):
        format = '.pdf'
        pdfFilePath = self.FigPath(name,format)

        savefig(
            pdfFilePath,
            dpi=300,
            bbox_inches='tight'
        )

        if self.PdfPopUp:
            sumatraPath = self.projectFolder/'.vscode'/'SumatraPDF.lnk'
            cmd = f'"{str(sumatraPath)}" "{str(pdfFilePath)}"'
            subprocess.Popen(cmd,shell=True)

    def SaveFig2mat(self,f):
        format = '.mat'
        self.matFilePath = self.FigPath(self.names[f],format)

        savemat(self.matFilePath,getattr(self,f'fig{f+1}'))

    def SaveFig2tex(self,f):
        format = '.tex'
        self.texFilePath = self.FigPath(self.names[f],format)

        cmd = self.cmdTeX()
        subprocess.run(cmd,shell=True)

    def CreateFolders(self,format):
        folderPath = self.projectFolder/'Figure'/format/self.regCode/self.folder
        Path(folderPath).mkdir(parents=True,exist_ok=True)
        # Define the folder path and create all the missing ones if necessary
        return folderPath

    def FigPath(self,figName,ext):
        return (getattr(self,f'{ext[1:]}FolderPath')/figName).with_suffix(ext)

    def cmdTeX(self):
        m = self.matFilePath.as_posix()
        t = self.texFilePath.as_posix()

        cmd = (
            r'matlab -batch "mat2tex('
            fr"'{m}',"fr"'{t}'"r')"'
        )
        return cmd

# Primary/Basic plots
def CreateFunctionPlot(
    x,y,
    figData,
    Ni=1,
    ta=None,
    yErr=True,
    label='',
    linestyle='-',
    linewidth=1,
    marker='None',
    hatch=None,
    color='black',
    alpha=1,
    idx='',
    ax=None
):
    if ax is None: ax = plt.gca()

    if Ni == 1:
        ax.plot(
            x,y if y.ndim == 1 else y.ravel(),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markevery=x.size//10-1,
            mfc=color,
            alpha=alpha
        )
        figData['plots'][f'functionPlot{idx}'] = {
            't':'function','x':x,'y':y,
            'l':label,'c':color,'m':marker
        }

    else:
        if yErr:
            avrFunc, err95Func = EvaluateConfidenceInterval(y,ta,Ni)

            # To avoid [a possible] lower negative confidence interval, the error has to be clipped
            lowerEstimate = np.maximum(avrFunc-err95Func,0)
            yerr95 = np.array([
                avrFunc-lowerEstimate,
                err95Func
            ]); xerr95 = None; y = avrFunc

            if (y-yerr95[0]).min()<0: raise("Negative values")
            if (y+yerr95[1]).min()<0: raise("Negative values")

        else:
            avrFunc, err95Func = EvaluateConfidenceInterval(x,ta,Ni)

            # To avoid [a possible] lower negative confidence interval, the error has to be clipped
            lowerEstimate = np.maximum(avrFunc-err95Func,0)
            xerr95 = np.array([
                avrFunc-lowerEstimate,
                err95Func
            ]); yerr95 = None; x = avrFunc

        ax.plot(
            x,y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markevery=x.size//10-1,
            mfc=color,
            alpha=alpha[0],
            zorder=4
        )
        figData['plots'][f'functionPlot{idx}'] = {
            't':'function','x':x,'y':y,
            'l':label,'c':color,'m':marker
        }

        CreateConfidenceIntervalPlot(
            x,y,
            figData,
            typ='functionfill',
            xerr95=xerr95,
            yerr95=yerr95,
            color=color,
            hatch=hatch,
            alpha=alpha[1],
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
        sct = ax.scatter(
            x,y if y.ndim == 1 else y.ravel(),
            label=label,
            color=color,
            s=size,
            edgecolor='none',
            alpha=alpha
        )
        figData['plots'][f'scatterPlot{idx}'] = {
            't':'scatter','x':x,'y':y,
            'l':label,'c':color,'a':alpha
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
                edgecolor='none',
                alpha=alpha[0],
                zorder=5
            )
            figData['plots'][f'scatterPlot{idx}'] = {
                't':'scatter','x':x,'y':avrSct,
                'l':label,'c':color,'a':alpha[0]
            }

            CreateConfidenceIntervalPlot(
                x,
                avrSct,
                figData,
                typ='errorbar',
                yerr95=np.array([err95Sct,err95Sct]),
                color=color,
                alpha=alpha[1],
                ax=ax,
                idx=idx
            )

        else:
            avrSct, err95Sct = EvaluateConfidenceInterval(x,ta,Ni)

            sct = ax.scatter(
                avrSct,
                y,
                label=label,
                color=color,
                s=size,
                edgecolor='none',
                alpha=alpha[0],
                zorder=5
            )
            figData['plots'][f'scatterPlot{idx}'] = {
                't':'scatter','x':avrSct,'y':y,
                'l':label,'c':color,'a':alpha[0]
            }

            CreateConfidenceIntervalPlot(
                avrSct,
                y,
                figData,
                typ='errorbar',
                xerr95=np.array([err95Sct,err95Sct]),
                color=color,
                alpha=alpha[1],
                ax=ax,
                idx=idx
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
    xScale='lin',
    Ni=1,
    ta=None,
    label='Histogram',
    color='#808080',
    norm=True,
    alpha=1,
    idx='',
    ax=None
):
    def HistogramPlot(
        binEdges,
        binMidPoints,
        binAverages,
        alpha=alpha
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
            't':'histogram','l':label,'c':color,'a':alpha,
            'x':binMidPoints,'b':binEdges,'w':binAverages
        }

        # hgPlot[0] = heights,
        # hgPlot[1] = bin edges,
        # hgPlot[2] = patches (Rectangle objects)

        # mplcursors.cursor(hgPlot[2],hover=False).connect(
        #     "add",lambda sel: sel.annotation.set_text(
        #         f"{hgPlot[0][sel.index]:.3f}"
        #     )
        # )

    if ax is None: ax = plt.gca()

    if limits is None:
        xMin = np.min(x); xMax=np.max(x)
    else:
        xMin = limits[0]; xMax=limits[1]

    if xScale == 'lin':
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
        binAverages, _ = np.histogram(
            x if x.ndim == 1 else x.ravel(),
            binEdges,density=norm
        )
        HistogramPlot(binEdges,binMidPoints,binAverages,alpha)

        return binMidPoints, binAverages
    else:
        hgData = [None]*Ni
        for r in range(Ni):
            hgData[r] = np.histogram(x[r,:],binEdges,density=norm)

        binAverages, hgErr95 = EvaluateConfidenceInterval(
            [hgData[r][0] for r in range(Ni)],ta,Ni
        )

        HistogramPlot(binEdges,binMidPoints,binAverages,alpha[0])

        # To avoid [a possible] lower negative confidence interval in the lower tail, the error has to be clipped
        lowerBinEstimate = np.maximum(binAverages-hgErr95,0)
        binErr95 = np.array([
            binAverages-lowerBinEstimate,
            hgErr95
        ])

        CreateConfidenceIntervalPlot(
            binMidPoints,
            binAverages,
            figData,
            typ='errorbar',
            yerr95=binErr95,
            color=color,
            ax=ax,
            idx=idx,
            alpha=alpha[1]
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
    limits=None,
    xScale='lin',
    Ni=1,
    ta=None,
    label='Lognormal fit (ML)', # Maximum likelihood
    color='black',
    alpha=1,
    marker='None',
    bimodal=False,
    idx='',
    ax=None
):
    if ax is None: ax=plt.gca()

    if limits is None:
        vMin = np.min(v); vMax=np.max(v)
    else:
        vMin = limits[0]; vMax=limits[1]

    if xScale == 'lin':
        xF = np.linspace(
            vMin,
            vMax,
            1000
        )
    else:
        xF = np.logspace(
            np.log10(vMin),
            np.log10(vMax),
            1000
        )

    yFData = [None]*Ni
    xiData = [None]*Ni

    for r in range(Ni):
        if bimodal:
            xi,scale1,shape1,scale2,shape2 = FitLognormalData(v[r,:],bimodal)
            yFData[r] = (
                xi*stats.lognorm.pdf(xF,s=shape1,scale=np.exp(scale1)) + 
                (1-xi)*stats.lognorm.pdf(xF,s=shape2,scale=np.exp(scale2))
            )
            xiData[r] = xi
        else:
            shape,loc,scale = FitLognormalData(v[r,:],bimodal)
            yFData[r] = stats.lognorm.pdf(xF,shape,loc=loc,scale=scale)
            xiData[r] = 1

    if Ni>1: yF = yFData; xi = xiData
    else: yF = yFData[0]; xi = xiData[0]

    # Fit plot
    CreateFunctionPlot(
        xF,yF,
        figData,
        Ni=Ni,
        ta=ta,
        label=label,
        # linewidth=1,
        color=color,
        alpha=alpha,
        marker=marker,
        idx=idx,
        ax=ax
    )

    return xi

def CreateParetoFitPlot(
    v,
    figData,
    upperbound=None,
    yScale='lin',
    Ni=1,
    ta=None,
    label=(
        'Scatter',
        'Pareto fit (ML)' # Minimum likelihood
    ),
    color=(
        'black',
        'black'
    ),
    alpha=1,
    bimodal=False,
    idx='',
    ax=None
):
    if ax is None: ax=plt.gca()

    if upperbound is None:
        xF = np.logspace(
            np.log10(np.quantile(v,0.75)),
            np.log10(v.max()),
            100
        )
    else:
        xF = np.logspace(
            np.log10(np.quantile(v,0.75)),
            np.log10(upperbound),
            100
        )

    vTails  = [None]*Ni
    b       = [None]*Ni
    ccdfFit = [None]*Ni

    for r in range(Ni):
        # Select the last quarter of city sizes
        vQuarter = np.quantile(v[r,:],.75)
        vTail = v[r,v[r,:] >= vQuarter]
        vTails[r] = np.sort(vTail) # Ascending values

        # Fitted CCDF from a Pareto function
        b[r], loc, scale = stats.pareto.fit(vTail,floc=0,fscale=vQuarter) # b≈alpha
        ccdfFit[r] = stats.pareto.sf(xF,b[r],loc=loc,scale=scale) # Survival function

    # Empirical CCDF on the tail
    n = vTails[0].size
    ccdfEmp = (
        n-np.arange(1,n+1,dtype=float)
        +(0.5 if yScale == 'log' else 0)
    )/n # P(X≥x)
    yS = ccdfEmp

    """
        The survival function is the Complementary Cumulative Distribution Function (CCDF)

        Formally it should be
        
                1-np.arange(1,n+1,dtype=float)/Nc-(Nc-n)/Nc=
                (n-np.arange(1,n+1,dtype=float))/Nc=
        (*)     [(n-np.arange(1,n+1,dtype=float))/n][n/Nc]
        
        hence the correct variables are:

            xS = vTails
            yS = (n-np.arange(1,n+1,dtype=float))/n*(n/Nc)
            yF = [ccdfFit[r]*(n/Nc) for r in range(Ni)]

        without the constant translation 0.5, which is just necessary to avoid having a zero value at the tail end in a log-log plot.

        However, since the only relevant information is the Pareto index, the exact probability value can be omitted: indeed, by dividing (*) by [n/Nc] the CCDF is rescaled in a linear scale while translated upward in a logarithmic one; both transformations do not affect the actual trend of the CCDF from which the Pareto index is calculated
    """

    if Ni>1:
        xS = vTails
        yF = ccdfFit
    else:
        xS = vTails[0]
        yF = ccdfFit[0]
        b = b[0]

    CreateScatterPlot(
        xS,yS,
        figData,
        Ni=Ni,
        ta=ta,
        size=10,
        yErr=False,
        label=label[0],
        color=color[0],
        alpha=alpha[0],
        idx=idx,
        ax=ax
    )

    CreateFunctionPlot(
        xF,yF,
        figData,
        Ni=Ni,
        ta=ta,
        label=label[1],
        color=color[1],
        alpha=alpha[1],
        marker='+' if bimodal else 'None',
        hatch='|' if bimodal else 'None',
        idx=idx,
        ax=ax
    )

    if bimodal:
        xFl = np.log(xF)
        yFData = [None]*Ni
        Nc = v.shape[1]

        for r in range(Ni):
            xi,loc1,shape1,loc2,shape2 = FitLognormalData(v[r,:] if Ni>1 else v,bimodal)

            ccdfFit1 = stats.norm.sf(xFl,loc=loc1,scale=shape1)
            ccdfFit2 = stats.norm.sf(xFl,loc=loc2,scale=shape2)

            yFData[r] = xi*ccdfFit1+(1-xi)*ccdfFit2
            # yFData[r] /= yFData[r][0] # Normalization
            yFData[r] /= n/Nc # Rescaling
            # It's necessary to effectively compare it with the Pareto fitting

        yF = yFData if Ni>1 else yFData[0]

        CreateFunctionPlot(
            xF,yF,
            figData,
            Ni=Ni,
            ta=ta,
            label=label[2],
            color=color[1],
            alpha=alpha[1],
            marker='x',
            hatch='/',
            idx=idx+1,
            ax=ax
        )

    return b


### Auxiliary functions ###

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
    x,y,
    figData,
    typ='errorbar',
    yerr95=None,
    xerr95=None,
    color='black',
    fmt='none',
    hatch=None,
    alpha=0.5,
    ax=None,
    idx=''
):
    if ax is None: ax = plt.gca()

    match typ:
        case 'errorbar':
            ax.errorbar(
                x,y,
                xerr=xerr95,
                yerr=yerr95,
                ecolor=color,
                fmt=fmt,
                linestyle='none',
                elinewidth=1.2,
                capsize=8,
                capthick=1.6,
                markersize=0,
                markerfacecolor=color,
                markerfacecoloralt=color,
                markeredgecolor=color,
                markeredgewidth=0,
                alpha=alpha,
                zorder=1
            )
            if xerr95 is None:
                figData['plots'][f'{typ}{idx}'] = {
                    't':typ,'x':x,'y':y,
                    'e':'y','ye':yerr95,
                    'l':'','c':color,'a':alpha,
                }
            else:
                figData['plots'][f'{typ}{idx}'] = {
                    't':typ,'x':x,'y':y,
                    'e':'x','xe':xerr95,
                    'l':'','c':color,'a':alpha,
                }

        case 'functionfill':
            if xerr95 is None:
                ax.fill_between(
                    x,y-yerr95[0],y+yerr95[1],
                    facecolor=color,
                    alpha=alpha,
                    linewidth=0,
                    hatch=hatch,
                    edgecolor=color,
                    zorder=2
                )
                figData['plots'][f'{typ}{idx}'] = {
                    't':typ,'x':x,'y':y,
                    'e':'y','ye':yerr95,
                    'l':'','c':color,'a':alpha,
                    'h':'None' if hatch is None else hatch,
                }
            else:
                ax.fill_betweenx(
                    y,x-xerr95[0],x+xerr95[1],
                    facecolor=color,
                    alpha=alpha,
                    linewidth=0,
                    hatch=hatch,
                    edgecolor=color,
                    zorder=2
                )
                figData['plots'][f'{typ}{idx}'] = {
                    't':typ,'x':x,'y':y,
                    'e':'x','xe':xerr95,
                    'l':'','c':color,'a':alpha,
                    'h':'None' if hatch is None else hatch,
                }

def FitLognormalData(v,bimodal):
    if not bimodal:
        shape,loc,scale = stats.lognorm.fit(v[v>0],floc=0)
        # The average is «μ=np.log(scale)» while the standard deviation is «σ=shape»
        return shape,loc,scale
    else:
        vp = v[v>0]
        vl = np.log(vp)

        # The objective function to minimize is the negative log-likelihood
        # This is equivalent to maximize the log-likelihood, just reversed
        def NegativeLogLikelihood(params):
            # xi is fraction between Gaussian
            # mi and si are the mean and standard deviation of the ith Gaussian
            xi,m1,s1,m2,s2 = params
            
            # Calculate Gaussian PDF for both components on log data
            pdf1 = stats.norm.pdf(vl,loc=m1,scale=s1)
            pdf2 = stats.norm.pdf(vl,loc=m2,scale=s2)
            
            # Mixture sum
            mixture = xi*pdf1+(1-xi)*pdf2
            
            # Prevent log(0) errors
            epsilon = 1e-10
            return -np.sum(np.log(mixture + epsilon))

        # Initial heuristic guesses
        m0 = np.mean(vl)
        s0 = np.std(vl)
        prm0 = [
            0.8, # The first gaussian is initially the most important one
            m0,s0,
            m0+0.5,s0
        ] # Same standard deviation but translated average

        bounds = ( # xi must be in [0,1]
            (0,1),       # xi
            (None,None), # m1
            (1e-5,None), # s1
            (None,None), # m2
            (1e-5, None) # s2
        ) # mi does not have a bound while si>0

        # Optimization
        minimization = minimize(NegativeLogLikelihood,prm0,bounds=bounds)
        
        # Extract results
        xi,m1,s1,m2,s2 = minimization.x

        return xi,m1,s1,m2,s2

def DataString(
    data,
    Ni=1,
    ta=None,
    head='',
    formatVal='.2f',
    formatErr='.2f',
    space=True
):
    (value,error) = EvaluateConfidenceInterval(data,ta,Ni)
    
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

#region From «FigData.SaveFig()» 
    """ Different naming rule between the folders
        match folder:
            case 'NA':
                def ext(figName,ext): return (folderFig/figName).with_suffix(ext)
            case 'KS':
                def ext(figName,ext):
                    figName = regCode['prefix']+figName
                    return (folderFig/figName).with_suffix(ext)
    """

    """ Open the pdf file (cross-platform)
        if sys.platform.startswith("win"):
            os.startfile(pdfFigPath)
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open",str(pdfFigPath)])
        else:
            subprocess.Popen(["xdg-open",str(pdfFigPath)])
    """

    """ Other formats
        pngFigPath  = ext(".png")
        htmlFigPath = ext(".html")
        plt.savefig(pngFigPath,dpi=300,bbox_inches='tight')
        mpld3.save_html(fig,str(htmlFigPath))
    """
#endregion

#region Old implementation of «CreateLognormalFitPlot» where the average was drawn over the lognormal fit as a vertical line
    """
        # In the function arguments
        label=(
            'Lognormal fit (ML)',
            'Average value'
        ), # Minimum likelihood
        color=(
            'blue',
            'black'
        ),

        # In the function body
        if Ni == 1:
            vAvr = np.mean(v)
            xM = np.array([vAvr,vAvr])
            yM = np.array([0,stats.lognorm.pdf(vAvr,shape,loc=loc,scale=scale)])
        else:
            avrData = [None]*Ni
            yM = 0

            for r in range(Ni):
                avrData[r] = np.mean(v[r,:])
                shape, loc, scale = stats.lognorm.fit(v[r,:],floc=0)
                yM += stats.lognorm.pdf(avrData[r],shape,loc=loc,scale=scale)/Ni

            xM = np.transpose(np.array([np.array(avrData)]*2))
            yM = np.array([0,yM])

        # Lognormal fit plot
        CreateFunctionPlot(
            xF,yF,
            figData,
            Ni=Ni,
            ta=ta,
            label=label[0],
            # linewidth=1,
            color=color[0],
            alpha=alpha,
            idx=idx,
            ax=ax
        )

        # Average value plot
        CreateFunctionPlot(
            xM,yM,
            figData,
            Ni=Ni,
            ta=ta,
            yErr=False,
            label=label[1],
            # linewidth=1,
            linestyle="--",
            color=color[1],
            idx=idx,
            ax=ax
        )
    """
    # In such a case the label would need to be a tuple
    # (
    #     fr'{lbl[t]} mean value $\langle k\rangle$',
    #     f'{lbl[t]} lognormal fit (ML)'
    # ),
    # (
    #     fr'{lbl[2]} mean value $\langle k\rangle$',
    #     f'{lbl[2]} lognormal fit (ML)'
    # ),
#endregion
