import tkinter as tk
from tkinter import ttk

global buttonFlat, dicObjects


### Main functions ###

def GUI():
    #region Global settings
    window = tk.Tk() # Main window
    window.title('City Size Model')
    window.resizable(False,False)  # Disable resizing the window

    pad = 40
    mainFrame = ttk.Frame() # Main frame
    mainFrame.grid(padx=pad,pady=pad/1.5)

    fontStyle = ('JetBrains Mono',15)
    # fontStyle = font.Font(family="JetBrains Mono", size=15)
    dicObjects = {
        # Sardinia population in 1991
        'totalPop':{ 'text':'S', 'value':int(1.6e6) },
        'timeStep':{ 'text':'Δt', 'value':1e-2 },
        'stepNumber':{ 'text':'Nt', 'value':int(2e4) },
        'iterations':{ 'text':'Ni', 'value':int(1) },
        'attractivity':{ 'text':'λ', 'value':.75 },
        'speed':{ 'text':'a', 'value':1 },
        'deviation':{ 'text':'σ', 'value':5e-2 },
        'checkbox':{ 'text':'Extract data', 'value':False }
    }
    #endregion

    #region Population parameters
    popPrmLabel = ttk.Label(
        mainFrame,
        text=f'Population parameters',
        font=fontStyle,
        anchor='s',
    )
    popPrmLabel.grid(pady=(0,10))

    popPrmFrame = ttk.Frame(mainFrame)
    popPrmFrame.grid()
    colSep = 15; rowSep = 5

    # First row
    row = 0
    dicObjects['attractivity']['obj'] = CreateLabelField(
        ttk.Entry,popPrmFrame,
        dicObjects['attractivity']['text'],
        dicObjects['attractivity']['value'],
        fontStyle,rI=row,cI=0,px=(0,colSep),py=(0,0)
    )
    dicObjects['speed']['obj'] = CreateLabelField(
        ttk.Entry,popPrmFrame,
        dicObjects['speed']['text'],
        dicObjects['speed']['value'],
        fontStyle,rI=0,cI=1,px=(0,0),py=(0,0)
    )

    # Second row
    row += 1
    dicObjects['totalPop']['obj'] = CreateLabelField(
        ttk.Entry,popPrmFrame,
        dicObjects['totalPop']['text'],
        dicObjects['totalPop']['value'],
        fontStyle,rI=row,cI=0,px=(0,colSep),py=(0,0)
    )
    dicObjects['deviation']['obj'] = CreateLabelField(
        tk.Scale,popPrmFrame,
        dicObjects['deviation']['text'],
        dicObjects['deviation']['value'],
        fontStyle,rI=row,cI=1,px=(0,0),py=(0,0),
        upperbound=1-dicObjects['attractivity']['obj']['var'].get()
    )

    dicObjects['attractivity']['obj']['var'].trace_add("write",DeviationUpperLimit)
    #endregion

    #region Time parameters
    timePrmLabel = ttk.Label(
        mainFrame,
        text=f'Time parameters',
        font=fontStyle,
        anchor='s',
    )
    timePrmLabel.grid(pady=(20,10))

    timePrmFrame = ttk.Frame(mainFrame)
    timePrmFrame.grid()
    colSep = 15

    # First row
    row = 0
    dicObjects['timeStep']['obj'] = CreateLabelField(
        ttk.Entry,timePrmFrame,
        dicObjects['timeStep']['text'],
        dicObjects['timeStep']['value'],
        fontStyle,rI=row,cI=0,px=(0,0),py=(0,rowSep)
    )
    row += 1
    dicObjects['stepNumber']['obj'] = CreateLabelField(
        ttk.Entry,timePrmFrame,
        dicObjects['stepNumber']['text'],
        dicObjects['stepNumber']['value'],
        fontStyle,rI=row,cI=0,px=(0,0),py=(0,rowSep)
    )
    row += 1
    dicObjects['iterations']['obj'] = CreateLabelField(
        ttk.Entry,timePrmFrame,
        dicObjects['iterations']['text'],
        dicObjects['iterations']['value'],
        fontStyle,rI=row,cI=0,px=(0,0),py=(0,0)
    )
    #endregion

    #region Checkbox
    style = ttk.Style()
    style.configure(
        'ckb.TCheckbutton',
        font=fontStyle
    )

    checkboxFrame = ttk.Frame(mainFrame)
    checkboxFrame.grid(pady=(20,0))

    inputVar = tk.BooleanVar(value=dicObjects['checkbox']['value'])
    inputCheckBox = ttk.Checkbutton(
        master=checkboxFrame,
        text=dicObjects['checkbox']['text'],
        variable=inputVar,
        style='ckb.TCheckbutton',
    )
    inputCheckBox.grid()
    dicObjects['checkbox']['obj'] = {
        'wid':inputCheckBox,
        'var':inputVar,
    }
    #endregion

    #region Buttons
    style = ttk.Style()
    style.configure(
        'btt.TButton',
        padding=(20,20),
        width=10,
        font=(fontStyle[0],15)
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
    CentrePlot(window)
    window.mainloop()

    dicParameters = {'runState':buttonFlag}
    for key in dicObjects:
        dicParameters[key] = dicObjects[key]['obj']['var'].get()
    return dicParameters


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

    if type(fieldValue) == float:
        fieldVar = tk.DoubleVar(value=fieldValue)
    elif type(fieldValue) == int:
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

def RunCallBack(window):
    global buttonFlag
    buttonFlag = 1
    window.destroy() # Close the window after the run button is pressed

def StopCallBack(window):
    global buttonFlag
    buttonFlag = 0
    window.destroy() # Close the window after stop button is pressed

def CentrePlot(window):
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
    l = dicObjects['attractivity']['obj']['var'].get()
    res = dicObjects['deviation']['obj']['wid']['resolution']
    dicObjects['deviation']['obj']['wid']['to'] = 1-l-res
