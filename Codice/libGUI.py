import tkinter as tk
from tkinter import ttk

# Window
window = tk.Tk()
window.title('City-Size Model')
window.geometry('500x300')


### Auxilary functions ###

def creatInputField(window,labelText,fontStyle):
    inputFrame = ttk.Frame(master=window)

    inputVar   = tk.DoubleVar()
    inputEntry = ttk.Entry(
        master=inputFrame,
        textvariable=inputVar
    )
    inputLabel = ttk.Label(
        master=inputFrame,
        text=labelText,
        font=fontStyle
    )

    inputLabel.pack(side='left',padx=(0,10))
    inputEntry.pack(side='left')
    inputFrame.pack(pady=5)

    return inputVar

def runFunc():
    print(inputs['S'].get())

def rerunFunc():
    print('rerun')

def stopFunc():
    print('stop')


# Title
lambdaLabel = ttk.Label(
    master=window,
    text='City Size Model',
    font = 'Calibri 14 bold'
)
lambdaLabel.pack()


# Input fields
fontStyle = 'Calibri 11'

labels = ['S:','Δt:','Nt:','λ:','σ:']
inputs = {}
for label in labels:
    inputs[label[:-1]] = creatInputField(window,label,fontStyle)


# Buttons
buttonFrame = ttk.Frame(master=window)

runButton = ttk.Button(
    master=buttonFrame,
    text='Run',
    command=runFunc
)
runButton.pack(side='left',padx=(0,5))
rerunButton = ttk.Button(
    master=buttonFrame,
    text='Redo',
    command=rerunFunc
)
rerunButton.pack(side='left',padx=(0,5))
stopButton = ttk.Button(
    master=buttonFrame,
    text='Stop',
    command=stopFunc
)
stopButton.pack(side='left')

buttonFrame.pack(pady=10)


# Run
window.mainloop()