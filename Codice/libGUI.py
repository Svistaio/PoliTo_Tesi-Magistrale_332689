
import tkinter as tk
from tkinter import ttk

import numpy as np

from multiprocessing import shared_memory

### Main class ###

class ParametersGUI(tk.Tk):
    def __init__(self):
        # Resets the default root before creating the GUI
        tk._default_root = None

        #region Window
        super().__init__() # Main window
        self.title('City Size Model')
        self.resizable(False,False) # Disable resizing the window

        self.bind('<Escape>',lambda e: self.SetSimulationState(False))
        self.bind('<space>',lambda e: self.SetSimulationState(True))
        #endregion

        #region Parameters
        self.regPopList = { # Italian region sizes in 1991
            'Piemonte':int(4302565),
            "Valle d'Aosta":int(115938),
            'Lombardia':int(8856074),
            'Trentino-Alto Adige':int(890360),
            'Veneto':int(4380797),
            'Friuli-Venezia Giulia':int(1197666),
            'Liguria':int(1676282),
            'Emilia-Romagna':int(3909512),
            'Toscana':int(3529946),
            'Umbria':int(811831),
            'Marche':int(1429205),
            'Lazio':int(5140371),
            'Abruzzo':int(1249054),
            'Molise':int(330900),
            'Campania':int(5630280),
            'Puglia':int(4031885),
            'Basilicata':int(610528),
            'Calabria':int(2070203),
            'Sicilia':int(4966386),
            'Sardegna':int(1648248),
            'Italia':int(56778031)
        } # See Table 6.1 on p. 488 of «ISTAT Popolazione e abitazioni 1991 {04-12-2025}.pdf»
        self.population = Parameter('S',int(1))

        self.attractivity = Parameter('λ',float(1))
        self.convincibility = Parameter('α',float(1))
        self.deviation = Parameter('σ',float(1))

        self.regNameList = [
            'Piemonte',
            "Valle d'Aosta",
            'Lombardia',
            'Trentino-Alto Adige',
            'Veneto',
            'Friuli-Venezia Giulia',
            'Liguria',
            'Emilia-Romagna',
            'Toscana',
            'Umbria',
            'Marche',
            'Lazio',
            'Abruzzo',
            'Molise',
            'Campania',
            'Puglia',
            'Basilicata',
            'Calabria',
            'Sicilia',
            'Sardegna',
            'Italia'
        ]
        self.region = Parameter(
            'Region selected',
            'region',
            list=self.regNameList
        )
        self.regCodeList = {
            r:i+1 for i,r in enumerate(self.regNameList)
        }

        self.zetaFraction = Parameter('ζ',float(1))

        self.timestep = Parameter('Δt',float(1))
        self.timesteps = Parameter('Nt',int(1))
        self.iterations = Parameter('Ni',int(1))
        self.progressBar = Parameter('Progress Bar',True)

        self.extraction = Parameter('Extract data',False)
        self.analysis = Parameter('Network analysis',False)
        self.edgeWeights = Parameter('Edge weights',False)

        self.intLawList = [
            'λ(rs^α)/(1+rs^α)',         # 0
            'λ(rsk/α)/(1+rsk/α)',       # 1
            '(1-ζ)efl_k/α+ζefs_k/α',    # 2
            'λ(rsk^α)/(1+rsk^α)',       # 3
            '(1-ζ)efl_k^α+ζefs_k^α',    # 4
            'λ[rsk/(1+rsk)]^α',         # 5
            '(1-ζ)[efl_k]^α+ζ[efs_k]^α' # 6
        ]
        self.interactingLaw = Parameter(
            'Interacting law',
            'law',
            list=self.intLawList
        )
        self.intLawCodeList = {
            r:i for i,r in enumerate(self.intLawList)
        }

        self.PdfPopUp = Parameter('Open PDF',False)
        self.LaTeXConversion = Parameter('LaTeX Conversion',False)
        self.screenshots = Parameter('Ns',int(1)) # Number of screenshots [not considering the initial state]
        self.smoothingFactor = Parameter('Sf',int(1))

        self.parametricStudy = Parameter('Parametric study',False)

        self.studiedPrmList = [
            self.attractivity.text,
            self.convincibility.text,
            self.zetaFraction.text
        ]
        self.studiedParameter = Parameter(
            'Studied parameter',
            'study',
            list=self.studiedPrmList
        )
        self.studiedPrmCodeList = {
            r:i for i,r in enumerate(self.studiedPrmList)
        }

        self.startValuePrmStudy = Parameter('Start',float(1))
        self.endValuePrmStudy = Parameter('End',float(1))
        self.numberPrmStudy = Parameter('Nv',int(1))

        self.simFlag = Parameter(var=tk.BooleanVar(value=False))
        #endregion

        #region Parameter frames
        pad = 20; pad = (pad,pad)
        mainFrame = Frame(self,pad=pad,sticky='nsew')
        pad = 15; pad = (1.25*pad,pad)

        popPrmFrame = Frame(
            mainFrame,
            pad=pad,
            title='Population parameters'
        )
        timePrmFrame = Frame(
            mainFrame,
            pad=pad,
            title='Time parameters'#,labelWidth=3
        )
        simPrmFrame = Frame(
            mainFrame,
            pad=pad,
            title='Simulation parameters'
        )
        ppcPrmFrame = Frame(
            mainFrame,
            pad=pad,
            title='Postprocessing parameters'
        )
        pspPrmFrame = Frame(
            mainFrame,
            pad=pad,
            title='Parametric study parameters',
            colSpan=mainFrame.nCol,
            nCol=5,
            removed=True
        ); self.pspPrmFrame = pspPrmFrame
        buttonFrame = Frame(
            mainFrame,
            pad=(pad[0],(1.5*pad[1],0)),
            colSpan=mainFrame.nCol
        )
        parametricStudyFrame = Frame(
            mainFrame,
            colSpan=mainFrame.nCol,
            sticky='sw'
        )
        #endregion

        #region Population parameters
        popPrmFrame.LabelSlider(
            self.attractivity,(0,1),0.01,
            extremes=(False,False)
        )
        popPrmFrame.LabelSlider(
            self.deviation,(0,1),0.001,
            extremes=(True,False)
        )
        self.SetDeviationUpperLimit()
        popPrmFrame.LabelEntry(self.convincibility)

        popPrmFrame.LabelEntry(self.population)
        self.attractivity.var.trace_add("write",self.SetDeviationUpperLimit)

        popPrmFrame.LabelEntry(self.zetaFraction,colSpan=2)

        popPrmFrame.LabelComboBox(self.region)
        self.region.var.trace_add("write",self.SetPopulation)
        #endregion

        #region Time parameters
        timePrmFrame.LabelEntry(self.timestep,colSpan=timePrmFrame.nCol)
        timePrmFrame.LabelEntry(self.timesteps,colSpan=timePrmFrame.nCol)
        timePrmFrame.LabelEntry(self.iterations,colSpan=timePrmFrame.nCol)
        timePrmFrame.CheckBox(self.progressBar)
        #endregion

        #region Simulation parameters
        simPrmFrame.CheckBox(self.extraction)
        simPrmFrame.CheckBox(self.analysis)
        simPrmFrame.CheckBox(self.edgeWeights)

        simPrmFrame.LabelComboBox(self.interactingLaw)
        self.interactingLaw.var.trace_add("write",self.InteractingLawCallBack)
        #endregion

        #region Postprocessing parameters
        ppcPrmFrame.CheckBox(self.PdfPopUp)
        ppcPrmFrame.CheckBox(self.LaTeXConversion)
        ppcPrmFrame.LabelSlider(
            self.screenshots,
            (1,self.timesteps.var.get()),
            1,
            colSpan=ppcPrmFrame.nCol
        )
        self.timesteps.var.trace_add(
            "write",lambda *args: self.SetSliderUpperLimit(
                self.screenshots,self.timesteps
            )
        )
        ppcPrmFrame.LabelSlider(
            self.smoothingFactor,
            (1,self.screenshots.var.get()),
            1,
            colSpan=ppcPrmFrame.nCol
        )
        self.screenshots.var.trace_add(
            "write",lambda *args: self.SetSliderUpperLimit(
                self.smoothingFactor,self.screenshots
            )
        )
        #endregion

        #region Parametric study frame
        pspPrmFrame.LabelComboBox(
            self.studiedParameter,
            colSpan=1,
            cbWdith=1,
            # pad=((5,0),(0,0))
        )
        self.studiedParameter.var.trace_add(
            'write',self.SetStudiedParameterState
        )
        width = 5
        pspPrmFrame.Label(Parameter('with'),pad=pspPrmFrame.pad)
        pspPrmFrame.LabelEntry(self.startValuePrmStudy,entryWidth=width)
        pspPrmFrame.LabelEntry(self.endValuePrmStudy,entryWidth=width)
        pspPrmFrame.LabelEntry(self.numberPrmStudy,entryWidth=width)

        parametricStudyFrame.CheckBox(self.parametricStudy)
        self.parametricStudy.var.trace_add(
            'write',self.ShowParametricStudyFrame
        )
        #endregion

        #region Case study list
        self.caseStudiesDict = {
            'Default':{
                # self.population:self.regPopList['Sardegna'],
                self.attractivity:5e-2,
                self.convincibility:5e-1,
                self.deviation:5e-2,
                self.region:self.regNameList[19],
                self.zetaFraction:1e-1,
                self.timestep:1e-2,
                self.timesteps:int(1e5),
                self.iterations:3,
                self.progressBar:False,
                self.extraction:False,
                self.analysis:False,
                self.edgeWeights:False,
                self.interactingLaw:self.intLawList[3],
                self.PdfPopUp:False,
                self.LaTeXConversion:False,
                self.screenshots:int(100),
                self.smoothingFactor:int(10),
                self.studiedParameter:self.studiedPrmList[1],
                self.parametricStudy:False,
                self.startValuePrmStudy:.1,
                self.endValuePrmStudy:1,
                self.numberPrmStudy:3
            },
            'λ(rsk/α)/(1+rsk/α)':{
                # self.population:self.regPopList['Sardegna'],
                self.attractivity:5e-2,
                # self.convincibility:4,
                self.deviation:5e-2,
                self.region:self.regNameList[19],
                # self.zetaFraction:1e-1,
                self.timestep:1e-2,
                self.timesteps:int(5e5),
                self.iterations:15,
                self.progressBar:True,
                self.extraction:False,
                self.analysis:False,
                self.edgeWeights:False,
                self.interactingLaw:self.intLawList[1],
                self.PdfPopUp:False,
                self.LaTeXConversion:False,
                # self.screenshots:int(100),
                # self.smoothingFactor:int(10),
                self.studiedParameter:self.studiedPrmList[0],
                self.parametricStudy:False,
                self.startValuePrmStudy:1,
                self.endValuePrmStudy:1,
                self.numberPrmStudy:1
            },
            '(1-ζ)efl_k/α+ζefs_k/':{
                # self.population:self.regPopList['Sardegna'],
                self.attractivity:5e-2,
                # self.convincibility:4,
                self.deviation:5e-2,
                self.region:self.regNameList[19],
                # self.zetaFraction:1e-1,
                self.timestep:1e-2,
                self.timesteps:int(1e7),
                self.iterations:15,
                self.progressBar:True,
                self.extraction:False,
                self.analysis:False,
                self.edgeWeights:False,
                self.interactingLaw:self.intLawList[2],
                self.PdfPopUp:False,
                self.LaTeXConversion:False,
                # self.screenshots:int(100),
                # self.smoothingFactor:int(10),
                self.studiedParameter:self.studiedPrmList[0],
                self.parametricStudy:False,
                self.startValuePrmStudy:1,
                self.endValuePrmStudy:1,
                self.numberPrmStudy:1
            },
            'λ(rsk^α)/(1+rsk^α)':{
                # self.population:self.population.var.get(),
                self.attractivity:5e-2,
                self.convincibility:3e-1,
                self.deviation:5e-2,
                self.region:self.regNameList[19],
                self.zetaFraction:1e-1,
                self.timestep:1e-2,
                self.timesteps:int(1e7),
                self.iterations:15,
                self.progressBar:True,
                self.extraction:False,
                self.analysis:False,
                self.edgeWeights:False,
                self.interactingLaw:self.intLawList[3],
                self.PdfPopUp:False,
                self.LaTeXConversion:False,
                # self.screenshots:int(100),
                # self.smoothingFactor:int(10),
                self.studiedParameter:self.studiedPrmList[0],
                self.parametricStudy:False,
                self.startValuePrmStudy:1,
                self.endValuePrmStudy:1,
                self.numberPrmStudy:1
            },
            'λ[rsk/(1+rsk)]^α':{
                # self.population:self.population.var.get(),
                self.attractivity:5e-2,
                self.convincibility:3e-1,
                self.deviation:5e-2,
                self.region:self.regNameList[19],
                self.zetaFraction:1e-1,
                self.timestep:1e-2,
                self.timesteps:int(1e7),
                self.iterations:15,
                self.progressBar:True,
                self.extraction:False,
                self.analysis:False,
                self.edgeWeights:False,
                self.interactingLaw:self.intLawList[5],
                self.PdfPopUp:False,
                self.LaTeXConversion:False,
                # self.screenshots:int(100),
                # self.smoothingFactor:int(10),
                self.studiedParameter:self.studiedPrmList[0],
                self.parametricStudy:False,
                self.startValuePrmStudy:1,
                self.endValuePrmStudy:1,
                self.numberPrmStudy:1
            },
        }
        self.caseStudy = Parameter(
            'Case studies',
            list(self.caseStudiesDict.keys())[0],
            list=list(self.caseStudiesDict.keys()),#[1:]
        )
        #endregion

        #region Buttons and case studies
        buttonFrame.LabelComboBox(
            self.caseStudy,
            colSpan=buttonFrame.nCol,
            pad=((0,0),(0,10))
        )
        self.caseStudy.var.trace_add("write",self.SetCaseStudy)

        buttonFrame.Button(
            'Run',lambda: self.SetSimulationState(True)
        )
        buttonFrame.Button(
            'Stop',lambda: self.SetSimulationState(False)
        )
        #endregion

        # GUI call
        self.SetCaseStudy()

        self.update_idletasks()
        CentreGUI(self)
        self.deiconify()

        self.mainloop()

    # Callbacks
    def SetDeviationUpperLimit(self,*args):
        try:
            l = self.attractivity.var.get()
            if isinstance(l,float):
                res = self.deviation.wid['resolution']
                if l >= 1:
                    l=1-res
                    self.attractivity.var.set(l)
                self.deviation.wid['to'] = 1-l-res
        except Exception:
            pass

    def SetPopulation(self,*args):
        nameReg = self.region.var.get()
        popReg = self.regPopList[nameReg]
        self.population.var.set(popReg)

    def SetConvincibility(self,*args):
        if self.intLawCodeList[self.interactingLaw.var.get()] in (1,2):
            l = self.attractivity.var.get()
            self.convincibility.var.set(np.round(l/.01-1,decimals=2))
        else:
            l = self.attractivity.var.get()
            self.convincibility.var.set(
                np.round(np.log10(.01/l)/np.log10(.5),decimals=2)
            )

    def SetSimulationState(self,state):
        self.simFlag.var.set(state)
        self.destroy() # Close the window after any button is pressed

    def InteractingLawCallBack(self,*args):
        if self.intLawCodeList[self.interactingLaw.var.get()] in (2,4,6):
            self.zetaFraction.frame.grid()
            self.studiedParameter.wid['values'] = self.studiedPrmList
        else:
            self.zetaFraction.frame.grid_remove()
            self.studiedParameter.wid['values'] = self.studiedPrmList[:-1]

        if self.intLawCodeList[self.interactingLaw.var.get()] in (1,2,5,6):
            self.EnableCallBack(self.attractivity,self.SetConvincibility)
        else:
            self.DisableCallBack(self.attractivity)

        self.ResizeWindow()

    def SetSliderUpperLimit(self,slider,ref):
        try:
            val = ref.var.get()
        except Exception:
            return
        slider.wid["to"] = val

    def ShowParametricStudyFrame(self,*args):
        frame = self.pspPrmFrame

        if self.parametricStudy.var.get(): frame.grid()
        else: frame.grid_remove()
        self.SetStudiedParameterState()

        self.ResizeWindow()

    def SetStudiedParameterState(self,*args):
        state = 'normal'
        self.attractivity.wid['state'] = state
        self.attractivity.wid['sliderrelief'] = 'raised'
        self.convincibility.wid['state'] = state
        self.zetaFraction.wid['state'] = state

        checked = self.parametricStudy.var.get()
        state = 'disabled' if checked else 'normal'
        value = self.studiedPrmCodeList[self.studiedParameter.var.get()]

        if checked:
            match value:
                case 0: 
                    self.attractivity.wid['state'] = state
                    self.attractivity.wid['sliderrelief'] = 'flat'
                case 1:
                    self.convincibility.wid['state'] = state
                case 2: 
                    self.zetaFraction.wid['state'] = state

    def SetCaseStudy(self,*args):
        caseStudy = self.caseStudy.var.get()
        for prm,val in self.caseStudiesDict[caseStudy].items():
            prm.var.set(val)

        self.InteractingLawCallBack()
        if self.intLawCodeList[self.interactingLaw.var.get()] in (1,2):
            self.SetConvincibility()
        self.ShowParametricStudyFrame()
        self.SetStudiedParameterState()

    # Functions
    def GatherParameters(self):
        parameters = {}
        for attribute,value in self.__dict__.items():
            if isinstance(value,Parameter):
                parameters[attribute] = value.var.get()
                # if value.var is not None:
                # setattr(self,name,value.var.get())
                # value.val = value.var.get()
        
        parameters = Parameters(**parameters)

        parameters.region = self.regCodeList[
            parameters.region
        ]
        parameters.interactingLaw = self.intLawCodeList[
            parameters.interactingLaw
        ]
        parameters.studiedParameter = self.studiedPrmCodeList[
            parameters.studiedParameter
        ]

        return parameters

    def ResizeWindow(self,center=True):
        self.update_idletasks()

        ww = self.winfo_reqwidth()
        wh = self.winfo_reqheight()

        if center:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()

            x = max(0,(sw-ww)//2)
            y = max(0,(sh-wh)//2)

            # self.after(100,lambda: self.geometry(f"{ww}x{wh}+{x}+{y}"))
            self.geometry(f"{ww}x{wh}+{x}+{y}")
        else:
            # self.after(100,lambda: self.geometry(f"{ww}x{wh}"))
            self.geometry(f"{ww}x{wh}")

    def EnableCallBack(self,prm,clb):
        if prm.cbid is None:
            prm.cbid = prm.var.trace_add("write",clb)

    def DisableCallBack(self,prm):
        if prm.cbid is not None:
            prm.var.trace_remove("write",prm.cbid)
            prm.cbid = None

class ProgressGUI(tk.Tk):
    def __init__(
        self,Ni,Nt,
        namep,namee,named,
        sid=None,Nv=None
    ):
        #region Window
        super().__init__()

        self.resizable(False,False)
        self.grid_rowconfigure(0,weight=1)
        self.grid_columnconfigure(0,weight=1)

        self.title("Simulation Progress")
        self.iconify() # Start minimized
        #endregion

        #region Parameters
        shmp = shared_memory.SharedMemory(name=namep); self.shmp = shmp
        shme = shared_memory.SharedMemory(name=namee); self.shme = shme
        shmd = shared_memory.SharedMemory(name=named); self.shmd = shmd

        self.progress = np.ndarray((Ni,),dtype=np.int64,buffer=shmp.buf)
        self.elapsed = np.ndarray((Ni,),dtype=np.float64,buffer=shme.buf)
        self.done = np.ndarray((Ni,),dtype=np.int8,buffer=shmd.buf)

        self.Ni = Ni
        self.Nt = Nt
        # self.d = [False]*Ni
        #endregion

        #region Frames
        shown = 10

        padmf = 20; pad = (padmf,padmf)
        if Ni>shown:
            mainFrame = Frame(
                self,
                nCol=2,
                pad=pad,
                normalFontStyle=('JetBrains Mono',11),
                sticky="nsew"
            )
            barsFrame = mainFrame.ScrollFrame(
                nCol=1,
                normalFontStyle=('JetBrains Mono',11)
            )
        else:
            mainFrame = Frame(
                self,
                nCol=1,
                pad=pad,
                normalFontStyle=('JetBrains Mono',11)
            )
            barsFrame = mainFrame

        padpb = 5; pad = (padpb,padpb)
        self.frames = []; self.bars = []; self.status = []
        for i in range(Ni):
            progressBarFrame = Frame(barsFrame,nCol=3,pad=pad)

            label = f"p{'0' if i+1<10 else ''}{i+1}" # Process
            setattr(self,label,Parameter(text=label+':'))
            progressBarFrame.Label(getattr(self,label),width=4)

            progressBarFrame.ProgressBar(getattr(self,label),Nt)

            status = f"s{'0' if i+1<10 else ''}{i+1}"
            setattr(self,status,Parameter(text=''))
            progressBarFrame.Label(
                getattr(self,status),
                width=60,
                pad=((0,0),(0,0)),
                anchor='c'
            )

            self.frames.append(progressBarFrame)
            self.bars.append(getattr(self,label).wid)
            self.status.append(getattr(self,status).lbl)

        padcf = padpb
        completionFrame = Frame(
            mainFrame,
            colSpan=mainFrame.nCol,
            # sticky='se',
            pad=((0,0),(padcf,0))
        )
        self.completion = Parameter(
            text=f'/{Ni}' if Nv is None else fr'/{Ni} {sid}/{Nv}'
        )
        completionFrame.Label(self.completion)

        if Ni>shown:
            self.update_idletasks()
            guiw = barsFrame.winfo_width()+2*padmf
            guih = (
                (progressBarFrame.winfo_height()+2*padpb)*shown
                +2*padmf+completionFrame.winfo_height()+padcf
            )
            self.geometry(f'{guiw}x{guih}')
        #endregion

        self.centered = False
        self.bind("<Map>",lambda e:self.CenterWhenActive())

        self.after(100,self.PoolInfo)

    def PoolInfo(self):
        for p in range(self.Ni):
            nt = int(self.progress[p])
            el = float(self.elapsed[p])

            if el != 0:
                self.bars[p]['value'] = nt
                ips = max(1,nt)/max(el,1e-10) # Iterations per second
                self.status[p]['text'] = (
                    f'{nt}/{self.Nt} ['
                    f'{TimeFormatter(nt/ips)}<'
                    f'{TimeFormatter((self.Nt-nt)/ips)} '
                    # f'[{nt/ips:.2f}<{(self.Nt-nt)/ips:.2f}, '
                    f'{ips:.2f}it/s]'
                )

        self.completion.lbl['text']=(
            f'{np.sum(self.done)}'+self.completion.text
        )

        if np.all(self.done):
            self.destroy()
        else:
            self.after(100,self.PoolInfo)

    def CenterWhenActive(self):
        if self.state() != "normal":
            return

        if not self.centered:
            # self.update_idletasks()
            CentreGUI(self)
            self.centered=True


### Auxiliary classes ###

class Frame(ttk.Frame):
    def __init__(
        self,parent,
        nCol=2,
        colSpan=1,
        title=None,
        sticky='n',
        pos=None,
        pad=((0,0),(0,0)),
        labelWidth=None,
        normalFontStyle=None,
        titleFontStyle=None,
        removed=False
    ):
        super().__init__(parent,borderwidth=0)

        self.columnconfigure(nCol)
        self.nCol = nCol

        self.cCol = 0
        self.cRow = 0

        self.pCol = [0,0]
        self.pRow = [0,0]
        self.pad = (self.pCol,self.pRow)

        self.labelWidth = labelWidth
        self.entryWidth = 13
        self.comboBoxWidth = 20

        if normalFontStyle is None:
            if isinstance(parent,Frame):
                self.normalFontStyle = parent.normalFontStyle
            else:
                self.normalFontStyle = ('JetBrains Mono',15)
        else:
            self.normalFontStyle = normalFontStyle

        if titleFontStyle is None:
            if isinstance(parent,Frame):
                self.titleFontStyle = parent.titleFontStyle
            else:
                self.titleFontStyle = ('JetBrains Mono',17,'bold')
        else:
            self.titleFontStyle = titleFontStyle

        if title is not None:
            self.SetTitle(title)

        if isinstance(parent,Frame):
            if pos is None:
                pos = (parent.cRow,parent.cCol)
            parent.NextColumn(colSpan)
        else:
            pos = (0,0) # Main frame position

        self.grid(
            row=pos[0],
            column=pos[1],
            columnspan=colSpan,
            padx=pad[0],
            pady=pad[1],
            sticky=sticky
        )
        if removed: self.grid_remove()

    # Functions
    def NextRow(self):
        self.cRow +=1

        if self.cRow == 0:
            self.pRow[0] = 0
        else:
            self.pRow[0] = 5

    def SetRow(self,row): self.cRow = row

    def NextColumn(self,colSpan=1): 
        if self.cCol+colSpan < self.nCol:
            self.cCol += colSpan
        else:
            self.cCol = 0
            self.NextRow()

        if self.cCol == 0:
            self.pCol[1] = 0
        else:
            self.pCol[1] = 10

    def SeteColumn(self,column): self.cCol = column

    def SetTitle(self,title):
        label = ttk.Label(
            self,
            text=f'{title}',
            font=self.titleFontStyle,
            anchor='s',
        )
        label.grid(
            row=self.cRow,
            column=self.cCol,
            columnspan=self.nCol,
            pady=(0,12.5)
        )
        self.NextRow()

    def CheckDataType(self,data):
        if isinstance(data.val,float):
            data.var = tk.DoubleVar(value=data.val)
        elif isinstance(data.val,int):
            data.var = tk.IntVar(value=data.val)
        elif isinstance(data.val,str):
            data.var = tk.StringVar(value=data.val)
        else:
            data.var = tk.BooleanVar(value=data.val)

    # Default widgets
    def Label(
        self,
        data,
        width=None,
        anchor='e',
        sticky=None,
        colSpan=1,
        pos=None,
        pad=((0,3),2)
    ):
        if width is None: width=self.labelWidth
        data.lbl = ttk.Label(
            master=self,
            text=data.text,
            font=self.normalFontStyle,
            anchor=anchor,
            width=width
        )

        row = self.cRow if pos is None else pos[0]
        col = self.cCol if pos is None else pos[1]

        data.lbl.grid(
            row=row,
            column=col,
            columnspan=colSpan,
            padx=pad[0],pady=pad[1],
            sticky=sticky
        )
        self.NextColumn(colSpan)

    def Entry(self,data,width=None):
        self.CheckDataType(data)

        if width is None: width = self.entryWidth

        data.wid = ttk.Entry(
            master=self,
            textvariable=data.var,
            font=self.normalFontStyle,
            width=width,
        )
        data.wid.grid(
            row=self.cRow,
            column=self.cCol,
            pady=2
        )
        self.NextColumn()
        
    def Slider(
        self,
        data,
        bounds,
        res,
        extremes
    ):
        # Measure the character length in pixels
        fieldInput = ttk.Entry(
            master=self,
            font=self.normalFontStyle,
            width=self.entryWidth
        )
        fieldInput.grid(row=0,column=1)
        l = fieldInput.winfo_reqwidth()
        fieldInput.destroy()

        # Create the slider widget
        self.CheckDataType(data)
        data.wid = tk.Scale(
            master=self,
            variable=data.var,
            from_=bounds[0]+(not extremes[0])*res,
            to=bounds[1]-(not extremes[0])*res,
            resolution=res,
            orient="horizontal",
            font=self.normalFontStyle,
            length=l
        )
        data.wid.grid(
            row=self.cRow,
            column=self.cCol,
            pady=2
        )
        self.NextColumn()

    def ComboBox(
        self,
        data,
        colSpan=1,
        state='readonly',
        width=None
    ):
        self.CheckDataType(data)

        if width is None: width = self.comboBoxWidth
        data.wid = ttk.Combobox(
            master=self,
            values=data.list,
            textvariable=data.var,
            state=state,
            font=self.normalFontStyle,
            width=width
        )
        data.wid.grid(
            row=self.cRow,
            column=self.cCol,
            columnspan=colSpan
        )
        self.NextColumn(colSpan)

    def CheckBox(self,data,colSpan=2):
        style = ttk.Style()
        style.configure(
            'ckb.TCheckbutton',
            font=self.normalFontStyle
        )
        
        self.CheckDataType(data)
        data.wid = ttk.Checkbutton(
            master=self,
            text=data.text,
            variable=data.var,
            style='ckb.TCheckbutton',
        )

        # frameWidth = self.winfo_width()
        data.wid.grid(
            row=self.cRow,
            column=self.cCol,
            columnspan=colSpan,
            # sticky='w',
            # padx=(frameWidth/3,0)
        )

        self.NextColumn(colSpan)

    def Button(self,text,callBack):
        style = ttk.Style()
        style.configure(
            'btt.TButton',
            padding=(20,20),
            width=10,
            font=self.normalFontStyle
        )
        button = ttk.Button(
            self,
            text=text,
            command=callBack,
            style="btt.TButton"
        )
        button.grid(
            row=self.cRow,
            column=self.cCol,
            padx=2
        )
        self.NextColumn()

    def ProgressBar(self,data,max,width=400):
        data.wid = ttk.Progressbar(
            master=self,
            maximum=max,
            length=width
        )
        data.wid.grid(
            row=self.cRow,
            column=self.cCol,
        )
        self.NextColumn()

    # Compound widgets
    def LabelEntry(
        self,data,
        colSpan=1,
        pos=None,
        lblWidth=None,
        pad=None,
        entryWidth=None,
    ):
        if lblWidth is None: lblWidth = self.labelWidth 
        if pad is None: pad = (self.pCol,self.pRow)
        
        data.frame = Frame(
            self,colSpan=colSpan,
            pos=pos,
            pad=pad,
            labelWidth=lblWidth
        )

        data.text += ':'
        data.frame.Label(data,lblWidth)
        data.text = data.text[:-1]
        data.frame.Entry(data,entryWidth)

    def LabelSlider(
        self,data,bounds,res,
        extremes=(True,True),
        pos=None,
        labelWidth=None,
        colSpan=1
    ):
        data.frame = Frame(
            self,
            colSpan=colSpan,
            pos=pos,
            pad=(self.pCol,self.pRow)
        )

        data.text += ':'
        data.frame.Label(
            data,
            width=labelWidth
        )
        data.text = data.text[:-1]
        data.frame.Slider(
            data,
            bounds,
            res,
            extremes
        )

    def LabelComboBox(
        self,
        data,
        colSpan=2,
        pos=None,
        # pad=((0,0),(10,0)),
        pad=None,
        cbWdith=None
    ):
        if pad is None: pad = (self.pCol,self.pRow)

        data.frame = Frame(
            self,
            pos=pos,
            colSpan=colSpan,
            pad=pad
        )

        data.text += ':'
        data.frame.Label(
            data,
            colSpan=colSpan,
            pad=((0,0),(0,0)),
            anchor='s'
        )
        data.text = data.text[:-1]
        data.frame.ComboBox(data,colSpan,width=cbWdith)

    def ScrollFrame(self,*args,**kwargs):
        canvas = tk.Canvas(self,highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self,
            orient='vertical',
            command=canvas.yview
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.grid(row=0,column=1,sticky="ns")
        canvas.grid(row=0,column=0,sticky="nsew")
        self.NextRow()

        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)

        frame = Frame(canvas,*args,**kwargs)

        window = canvas.create_window(
            (0,0),
            window=frame,
            anchor='nw'
        )

        frame.bind(
            "<Configure>",
            lambda e:self.FrameConfiguration(canvas)
        )
        canvas.bind(
            "<Configure>",
            lambda e:self.CanvasConfiguration(e,canvas,window)
        )

        canvas.bind_all("<MouseWheel>",lambda e:self.ScrollMotion(e,canvas))

        return frame

    # CallBacks
    def FrameConfiguration(self,canvas):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def CanvasConfiguration(self,event,canvas,window):
        canvas.itemconfig(window,width=event.width)

    def ScrollMotion(self,event,canvas):
        canvas.yview_scroll(int(-event.delta/120),"units")


class Parameter():
    def __init__(
        self,
        text=None,
        val=None,
        lbl=None,
        var=None,
        wid=None,
        frame=None,
        list=None,
        cbid=None # CallBack id
    ):
        self.text = text
        self.val  = val
        self.lbl  = lbl
        self.var  = var
        self.wid  = wid
        self.frame = frame
        self.list = list
        self.cbid = cbid

class Parameters():
    def __init__(self,**kwargs):
        for text,value in kwargs.items():
            setattr(self,text,value)

def CentreGUI(window):
    # Get screen width and height
    sw = window.winfo_screenwidth()
    sh = window.winfo_screenheight()

    # Update window size
    window.update_idletasks()

    # Measure window size
    ww = window.winfo_width()
    wh = window.winfo_height()

    # Calculate new centred coordinates
    x = max(0,(sw-ww)//2)
    y = max(0,(sh-wh)//2)

    # Set geometry
    window.geometry(f"{ww}x{wh}+{x}+{y}")

def TimeFormatter(seconds):
    m, s = divmod(seconds,60)
    h, m = divmod(m,60)
    return f"{h:.0f}:{m:.0f}:{s:.0f}"


### Discarded code ###

#region Old and worse GUI implementation without classes
"""
    def GUI():
        global dicObjects, dicReg

        # Resets the default root before creating the GUI
        tk._default_root = None

        #region Global settings
        window = tk.Tk() # Main window
        window.title('City Size Model')
        window.resizable(False,False)  # Disable resizing the window

        pad = 40
        mainFrame = ttk.Frame(window) # Main frame
        mainFrame.grid(padx=pad,pady=pad/1.5)

        dicLayout = {
            'normalFontStyle':('JetBrains Mono',15),
            'groupFontStyle':('JetBrains Mono',17,'bold'),
            'groupVerSep':(30,5),
            'groupColSep':15,
            'groupRowSep':5
        }

        dicReg = {
            'nameList':[
                'Piemonte',
                "Valle d'Aosta",
                'Lombardia',
                'Trentino-Alto Adige',
                'Veneto',
                'Friuli-Venezia Giulia',
                'Liguria',
                'Emilia-Romagna',
                'Toscana',
                'Umbria',
                'Marche',
                'Lazio',
                'Abruzzo',
                'Molise',
                'Campania',
                'Puglia',
                'Basilicata',
                'Calabria',
                'Sicilia',
                'Sardegna',
                'Italia'
            ],
            'popList':{ # Italian region sizes in 1991
                'Piemonte':int(4302565),
                "Valle d'Aosta":int(115938),
                'Lombardia':int(8856074),
                'Trentino-Alto Adige':int(890360),
                'Veneto':int(4380797),
                'Friuli-Venezia Giulia':int(1197666),
                'Liguria':int(1676282),
                'Emilia-Romagna':int(3909512),
                'Toscana':int(3529946),
                'Umbria':int(811831),
                'Marche':int(1429205),
                'Lazio':int(5140371),
                'Abruzzo':int(1249054),
                'Molise':int(330900),
                'Campania':int(5630280),
                'Puglia':int(4031885),
                'Basilicata':int(610528),
                'Calabria':int(2070203),
                'Sicilia':int(4966386),
                'Sardegna':int(1648248),
                'Italia':int(56778031)
            } # See Table 6.1 on p. 488 of «ISTAT Popolazione e abitazioni 1991 {04-12-2025}.pdf»
        }
        dicReg['codeList'] = {r:int(i+1) for i,r in enumerate(dicReg['nameList'])}

        dicObjects = {
            'totalPop':{ 'text':'S', 'value':dicReg['popList']['Sardegna'] },
            'attractivity':{ 'text':'λ', 'value':.05 },
            'convincibility':{ 'text':'α', 'value':4 },
            'deviation':{ 'text':'σ', 'value':5e-2 },
            'regSelected':{ 'text':'Region selected', 'value':dicReg['nameList'][19] },
            #
            'timeStep':{ 'text':'Δt', 'value':1e-2 },
            'stepNumber':{ 'text':'Nt', 'value':int(2e5) },
            'iterations':{ 'text':'Ni', 'value':int(1) },
            #
            'extraction':{ 'text':'Extract data', 'value':False },
            'analysis':{ 'text':'Network analysis', 'value':False },
            'edgeWeights':{ 'text':'Edge weights', 'value':False },
        }
        #endregion

        #region Population parameters
        popPrmLabel = ttk.Label(
            mainFrame,
            text=f'Population parameters',
            font=dicLayout['groupFontStyle'],
            anchor='s',
        )
        popPrmLabel.grid(pady=(0,dicLayout['groupVerSep'][1]))

        popPrmFrame = ttk.Frame(mainFrame)
        popPrmFrame.grid()

        # First row
        row = 0
        dicObjects['attractivity']['obj'] = CreateLabelField(
            ttk.Entry,popPrmFrame,
            dicObjects['attractivity']['text'],
            dicObjects['attractivity']['value'],
            dicLayout['normalFontStyle'],
            rI=row,cI=0,px=(0,dicLayout['groupColSep']),py=(0,0)
        )
        dicObjects['convincibility']['obj'] = CreateLabelField(
            ttk.Entry,popPrmFrame,
            dicObjects['convincibility']['text'],
            dicObjects['convincibility']['value'],
            dicLayout['normalFontStyle'],
            rI=0,cI=1,px=(0,0),py=(0,0)
        )

        dicObjects['attractivity']['obj']['var'].trace_add("write",DeviationUpperLimit)

        # Second row
        row += 1
        dicObjects['totalPop']['obj'] = CreateLabelField(
            ttk.Entry,popPrmFrame,
            dicObjects['totalPop']['text'],
            dicObjects['totalPop']['value'],
            dicLayout['normalFontStyle'],
            rI=row,cI=0,px=(0,dicLayout['groupColSep']),py=(0,0)
        )
        dicObjects['deviation']['obj'] = CreateLabelField(
            tk.Scale,popPrmFrame,
            dicObjects['deviation']['text'],
            dicObjects['deviation']['value'],
            dicLayout['normalFontStyle'],
            rI=row,cI=1,px=(0,0),py=(0,0),
            upperbound=1-dicObjects['attractivity']['obj']['var'].get()
        )


        regLabel = ttk.Label(
            master=mainFrame,
            text=f'{dicObjects['regSelected']['text']}:',
            font=dicLayout['normalFontStyle'],
            anchor='s'
        )
        regLabel.grid(pady=(10,0))

        stringVar = tk.StringVar(value=dicObjects['regSelected']['value'])
        intVar = tk.IntVar(value=dicReg['codeList'][stringVar.get()])
        regComboBox = ttk.Combobox(
            master=mainFrame,
            values=dicReg['nameList'],
            textvariable=stringVar,
            font=dicLayout['normalFontStyle'],
            width=20
        )
        regComboBox.grid()
        dicObjects['regSelected']['obj'] = {
            'wid':regComboBox,
            'var':intVar,
            'lab':stringVar
        }
        dicObjects['regSelected']['obj']['lab'].trace_add("write",ChangePopulation)
        #endregion

        #region Time parameters
        timePrmLabel = ttk.Label(
            mainFrame,
            text=f'Time parameters',
            font=dicLayout['groupFontStyle'],
            anchor='s',
        )
        timePrmLabel.grid(pady=dicLayout['groupVerSep'])

        timePrmFrame = ttk.Frame(mainFrame)
        timePrmFrame.grid()

        # First row
        col = 0
        dicObjects['timeStep']['obj'] = CreateLabelField(
            ttk.Entry,timePrmFrame,
            dicObjects['timeStep']['text'],
            dicObjects['timeStep']['value'],
            dicLayout['normalFontStyle'],
            rI=0,cI=col,px=(0,dicLayout['groupColSep']),py=(0,0)
        )
        col += 1
        dicObjects['stepNumber']['obj'] = CreateLabelField(
            ttk.Entry,timePrmFrame,
            dicObjects['stepNumber']['text'],
            dicObjects['stepNumber']['value'],
            dicLayout['normalFontStyle'],
            rI=0,cI=col,px=(0,0),py=(0,0)
        )
        #endregion

        #region Simulation parameters
        style = ttk.Style()
        style.configure(
            'ckb.TCheckbutton',
            font=dicLayout['normalFontStyle']
        )

        simPrmLabel = ttk.Label(
            mainFrame,
            text=f'Simulation parameters',
            font=dicLayout['groupFontStyle'],
            anchor='s',
        )
        simPrmLabel.grid(pady=dicLayout['groupVerSep'])

        simPrmFrame = ttk.Frame(mainFrame)
        simPrmFrame.grid()

        dicObjects['iterations']['obj'] = CreateLabelField(
            ttk.Entry,simPrmFrame,
            dicObjects['iterations']['text'],
            dicObjects['iterations']['value'],
            dicLayout['normalFontStyle'],
            rI=0,cI=0,px=(0,0),py=(0,0)
        )

        dicObjects['extraction']['obj'] = CreateCheckBox(
            simPrmFrame,'ckb.TCheckbutton',
            dicObjects['extraction']['text'],
            dicObjects['extraction']['value']
        )
        dicObjects['analysis']['obj'] = CreateCheckBox(
            simPrmFrame,'ckb.TCheckbutton',
            dicObjects['analysis']['text'],
            dicObjects['analysis']['value']
        )
        dicObjects['edgeWeights']['obj'] = CreateCheckBox(
            simPrmFrame,'ckb.TCheckbutton',
            dicObjects['edgeWeights']['text'],
            dicObjects['edgeWeights']['value']
        )
        #endregion

        #region Buttons
        style = ttk.Style()
        style.configure(
            'btt.TButton',
            padding=(20,20),
            width=10,
            font=(dicLayout['normalFontStyle'][0],15)
        )

        buttonFrame = ttk.Frame(mainFrame)
        buttonFrame.grid(pady=(30,0))

        runButton = ttk.Button(
            buttonFrame,text='Run',
            command=lambda: RunCallBack(window),
            style="btt.TButton"
        )
        runButton.grid(row=0,column=0,padx=(0,5))

        stopButton = ttk.Button(
            buttonFrame,text='Stop',
            command=lambda: StopCallBack(window),
            style="btt.TButton"
        )
        stopButton.grid(row=0,column=1)
        #endregion

        # Run
        CentreGUI(window)
        window.mainloop()

        #region Output
        dicPrm = {'runState':buttonFlag}
        for key in dicObjects:
            dicPrm[key] = dicObjects[key]['obj']['var'].get()

        code = dicPrm['regSelected']
        libFigures.regCode = f'{'0' if code<=9 else ''}{code}'

        return dicPrm
        #endregion

    def CreateLabelField(
        fieldType,frame,
        fieldText,fieldValue,
        fontStyle,px,py,
        rI=None,cI=None,
        upperbound=None
    ):
        fieldFrame = ttk.Frame(frame)
        fieldFrame.grid(
            row=rI,column=cI,
            padx=px,pady=py
        )

        if isinstance(fieldValue,float):
            fieldVar = tk.DoubleVar(value=fieldValue)
        elif isinstance(fieldValue,int):
            fieldVar = tk.IntVar(value=fieldValue)

        fieldLabel = ttk.Label(
            master=fieldFrame,
            text=f'{fieldText}:',
            font=fontStyle,
            anchor='e',
            width=3
        )
        fieldLabel.grid(
            row=0,column=0,
            pady=2,padx=(0,3),
            sticky='e'
        )

        match fieldType:
            case ttk.Entry:
                fieldInput = fieldType(
                    master=fieldFrame,
                    textvariable=fieldVar,
                    font=fontStyle,
                    width=13,
                    # command=callBack
                )
            case tk.Scale:
                # Measure the character length in pixels
                fieldInput = ttk.Entry(
                    master=fieldFrame,
                    font=fontStyle,
                    width=13
                )
                fieldInput.grid(row=0,column=1)
                l = fieldInput.winfo_reqwidth()
                fieldInput.destroy()

                # Create the slider widget
                res = 0.001
                fieldInput = fieldType(
                    master=fieldFrame,
                    variable=fieldVar,
                    from_=0,
                    to=upperbound-res,
                    resolution=res,
                    orient="horizontal",
                    font=fontStyle,
                    length=l
                )

        fieldInput.grid(
            row=0,column=1,
            pady=2
        )

        dicRtr = {
            'wid':fieldInput,
            'var':fieldVar,
            'lab':fieldLabel,
        }
        return dicRtr

    def CreateCheckBox(
        frame,style,
        fieldText,fieldValue,
    ):
        fieldVar = tk.BooleanVar(value=fieldValue)
        fieldCheckBox = ttk.Checkbutton(
            master=frame,
            text=fieldText,
            variable=fieldVar,
            style=style,
        )
        fieldCheckBox.grid()

        dicRtr = {
            'wid':fieldCheckBox,
            'var':fieldVar,
        }
        return dicRtr

    def RunCallBack(window):
        global buttonFlag
        buttonFlag = 1
        window.destroy() # Close the window after the run button is pressed

    def StopCallBack(window):
        global buttonFlag
        buttonFlag = 0
        window.destroy() # Close the window after stop button is pressed

    def CentreGUI(window):
        # Get screen width and height
        screenWidth = window.winfo_screenwidth()
        screenHeight = window.winfo_screenheight()

        # Update window size
        window.update_idletasks()

        # Measure window size
        windowWidth = window.winfo_width()
        windowHeight = window.winfo_height()

        # Calculate new centred coordinates
        x = (screenWidth // 2) - (windowWidth // 2)
        y = (screenHeight // 2) - (windowHeight // 2)

        # Set geometry
        window.geometry(f"{windowWidth}x{windowHeight}+{x}+{y}")

    def DeviationUpperLimit(*args):
        global dicObjects
        try:
            l = dicObjects['attractivity']['obj']['var'].get()
            if type(l) == float:
                res = dicObjects['deviation']['obj']['wid']['resolution']
                if l >= 1:
                    l=1-res
                    dicObjects['attractivity']['obj']['var'].set(l)
                dicObjects['deviation']['obj']['wid']['to'] = 1-l-res
        except Exception:
            return

    def ChangePopulation(*args):
        global dicObjects, dicReg
        nameReg = dicObjects['regSelected']['obj']['lab'].get()

        codeReg = dicReg['codeList'][nameReg]
        popReg = dicReg['popList'][nameReg]

        dicObjects['totalPop']['obj']['var'].set(popReg)
        dicObjects['regSelected']['obj']['var'].set(codeReg)
"""
#endregion

#region Old implementation to show an arbitrary number of progress bars where only the bars associated to completed process would be hidden if there were still hidden bars of uncompleted processes which would take their place
"""
class ProgressGUI(tk.Tk):
    def __init__(
        self,Ni,Nt,
        namep,namee,named
    ):
        #region Window
        super().__init__()
        self.resizable(False,False)
        self.title("Simulation Progress")
        self.iconify() # Start minimized
        #endregion

        #region Parameters
        shmp = shared_memory.SharedMemory(name=namep); self.shmp = shmp
        shme = shared_memory.SharedMemory(name=namee); self.shme = shme
        shmd = shared_memory.SharedMemory(name=named); self.shmd = shmd

        self.progress = np.ndarray((Ni,),dtype=np.int64,buffer=shmp.buf)
        self.elapsed = np.ndarray((Ni,),dtype=np.float64,buffer=shme.buf)
        self.done = np.ndarray((Ni,),dtype=np.int8,buffer=shmd.buf)

        self.Ni = Ni
        self.Nt = Nt
        # self.d = [False]*Ni
        #endregion

        #region Frames
        pad = 20; pad = (pad,pad)
        mainFrame = Frame(
            self,
            nCol=1,
            pad=pad,
            normalFontStyle=('JetBrains Mono',11)
        )
        pad = 5; pad = (pad,pad)

        self.frames = []; self.bars = []; self.status = []
        for i in range(Ni):
            progressBarFrame = Frame(mainFrame,nCol=3,pad=pad)

            label = f"p{'0' if i+1<10 else ''}{i+1}" # Process
            setattr(self,label,Parameter(text=label+':'))
            progressBarFrame.Label(getattr(self,label),width=4)

            progressBarFrame.ProgressBar(getattr(self,label),Nt)

            status = f"s{'0' if i+1<10 else ''}{i+1}"
            setattr(self,status,Parameter(text=''))
            progressBarFrame.Label(
                getattr(self,status),
                width=60,
                pad=((0,0),(0,0)),
                anchor='c'
            )

            self.frames.append(progressBarFrame)
            self.bars.append(getattr(self,label).wid)
            self.status.append(getattr(self,status).lbl)

        progressBarFrame = Frame(mainFrame,sticky='e')
        self.completion = Parameter(text=f'0/{Ni}',val=0)
        progressBarFrame.Label(self.completion)

        shown = 10; self.shown = shown
        if Ni>shown:
            for i in range(Ni-shown):
                self.frames[shown+i].grid_remove()
        self.removed = [False]*Ni
        #endregion

        CentreGUI(self)
        self.after(100,self.PoolInfo)

    def PoolInfo(self):
        for p in range(self.Ni):
            nt = int(self.progress[p])
            el = float(self.elapsed[p])

            if el != 0:
                self.bars[p]['value'] = nt
                ips = max(1,nt)/max(el,1e-10) # Iterations per second
                self.status[p]['text'] = (
                    f'{nt}/{self.Nt} ['
                    f'{TimeFormatter(nt/ips)}<'
                    f'{TimeFormatter((self.Nt-nt)/ips)} '
                    # f'[{nt/ips:.2f}<{(self.Nt-nt)/ips:.2f}, '
                    f'{ips:.2f}it/s]'
                )

            if self.done[p] and not self.removed[p]:
                nextFrame = self.shown+self.completion.val
                if nextFrame<self.Ni:
                    self.frames[nextFrame].grid()
                    self.frames[p].grid_remove()

                self.completion.val+=1
                self.removed[p] = True

                self.completion.lbl['text']=f'{self.completion.val}/{self.Ni}'

        if np.all(self.done):
            self.destroy()
        else:
            self.after(100,self.PoolInfo)

"""
#endregion It was replaced by a much more simple approach with a scrollable frame
