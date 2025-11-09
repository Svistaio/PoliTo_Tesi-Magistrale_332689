import tkinter as tk
from tkinter import ttk

global flagButton
flagButton = 0


### Main functions ###

def GUI():
    # Main window
    window = tk.Tk()
    window.title('City Size Model')
    window.resizable(False,False)  # Disable resizing the window

    # Main frame
    pad = 40
    mainFrame = ttk.Frame() # window,padding=pad
    mainFrame.grid(padx=pad,pady=pad/1.5)


    # # Title
    # titleLabel = ttk.Label(
    #     mainFrame,
    #     text='City Size Model',
    #     font = 'Calibri 14 bold'
    # )
    # titleLabel.grid(
    #     row=0,column=0,
    #     pady=(0,10)
    # )


    # Input fields
    fontStyle = 'Calibri 11'

    dicLabels = {
        'sMax':[int(1.6e6),'S'], # Sardinia population in 1991
        'dt':[1e-2,'Δt'],
        'Nt':[int(2e4),'Nt'],
        'l':[.75,'λ'],
        'a':[1,'a'],
        'sigma':[5e-2,'σ']
    }
    dicInputs = {}

    labelFrame = ttk.Frame(mainFrame)
    labelFrame.grid()
    for i,label in enumerate(dicLabels):
        dicInputs[label] = CreateInputField(
            labelFrame,
            dicLabels[label][1],
            dicLabels[label][0],
            fontStyle,i
        )
    checkboxFrame = ttk.Frame(mainFrame)
    checkboxFrame.grid()

    inputVar = tk.BooleanVar(value=False)
    inputCheckBox = ttk.Checkbutton(
        master=checkboxFrame,
        text='Extract data',
        variable=inputVar
    )
    inputCheckBox.grid(row=0,column=0,pady=(10,0))
    dicInputs['extData'] = inputVar


    # Buttons
    buttonFrame = ttk.Frame(mainFrame)
    buttonFrame.grid(pady=(20,0))

    runButton = ttk.Button(
        buttonFrame,text='Run',
        command=lambda: RunCallBack(window)
    )
    runButton.grid(row=0,column=0,padx=(0,5))

    stopButton = ttk.Button(
        buttonFrame,text='Stop',
        command=lambda: StopCallBack(window)
    )
    stopButton.grid(row=0,column=1)


    # Run
    CentrePlot(window)
    window.mainloop()

    global flagButton
    dicOutput = {'runState':flagButton}
    if flagButton:
        for key in dicInputs:
            dicOutput[key] = dicInputs[key].get()
        return dicOutput
    else:
        return dicOutput


### Auxilary functions ###

def CreateInputField(frame,labelText,labelValue,fontStyle,i):
    if type(labelValue) == float:
        inputVar = tk.DoubleVar(value=labelValue)
    elif type(labelValue) == int:
        inputVar = tk.IntVar(value=labelValue)

    inputLabel = ttk.Label(
        frame,
        text=f'{labelText}:',
        font=fontStyle,
        anchor='e',
        width=5
    )
    inputEntry = ttk.Entry(
        frame,
        textvariable=inputVar,
        width=20
    )
    inputLabel.grid(row=i,column=0,pady=2,padx=(0,3),sticky='e')
    inputEntry.grid(row=i,column=1,pady=2)

    return inputVar

def RunCallBack(window):
    global flagButton
    flagButton = 1
    window.destroy() # Close the window after the run button is pressed

def StopCallBack(window):
    global flagButton
    flagButton = 0
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