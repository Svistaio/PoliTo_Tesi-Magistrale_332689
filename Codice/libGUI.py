import tkinter as tk
from tkinter import ttk

# Window
window = tk.Tk()
window.title('City-Size Model')
window.geometry('300x150')

# Title
titleLabel = ttk.Label(
    master=window,
    text='λ',
    font = 'Calibri 14 bold'
)
titleLabel.pack()

# Input field
inputFrame = ttk.Frame(master=window)
entry = ttk.Entry(master=inputFrame)
button = ttk.Button(
    master=inputFrame,
    text='Run'
)
entry.pack(); button.pack(); inputFrame.pack()

# Run
window.mainloop()