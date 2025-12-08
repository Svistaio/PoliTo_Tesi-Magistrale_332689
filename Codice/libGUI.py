import tkinter as tk
from tkinter import ttk

import libFigures

global buttonFlag
buttonFlat = False


### Main functions ###

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
        'attractivity':{ 'text':'λ', 'value':.75 },
        'convincibility':{ 'text':'α', 'value':1 },
        'deviation':{ 'text':'σ', 'value':5e-2 },
        'regSelected':{ 'text':'Region selected', 'value':dicReg['nameList'][19] },
        #
        'timeStep':{ 'text':'Δt', 'value':1e-2 },
        'stepNumber':{ 'text':'Nt', 'value':int(2e4) },
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


### Auxilary functions ###

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
