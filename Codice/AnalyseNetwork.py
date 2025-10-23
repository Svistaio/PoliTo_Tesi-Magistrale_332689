import os

from zipfile import ZipFile as ZF
import io

import numpy as np
from scipy.stats import lognorm

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import mplcursors

def centrePlot():
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

A = np.array([],dtype=int)

with ZF("../Dati/AdjacencyMatricesIt91.zip") as z:
    with z.open("20AdjacencyMatrixSardegna.txt","r") as f:
        A = np.loadtxt( # Decodes binary stream as UTF-8 text
            io.TextIOWrapper(f,encoding="utf-8"),
            delimiter=",",
            dtype=int
        )

# Vectors of degrees
d = np.sum(A, axis=0)
N = len(d) # Normalization factor

# Unique degrees and corresponding frequencies
k, counts = np.unique(d,return_counts=True)
Pk = counts/N

shape, loc, scale = lognorm.fit(k,floc=0)
x = np.linspace(min(k),max(k),500)
pdf = lognorm.pdf(x,shape,loc=loc,scale=scale)

sc = plt.scatter(k,Pk,label="Empirical",s=10)
plt.plot(
    x,pdf, # /np.sum(pdf)*np.sum(Pk)
    label="Lognormal fit",
    color="red"
)

plt.xlabel(r"$k$")
plt.ylabel(r"$P(k)$")

plt.xlim(0,300)
plt.ylim(0,0.04)

plt.grid(True,linestyle=":",linewidth=1)
plt.legend()

mplcursors.cursor(sc,hover=True).connect(
    "add",lambda sel: sel.annotation.set_text(
        f"k={k[sel.index]}, P(k)={Pk[sel.index]:.3f}"
    )
)

centrePlot()

path = "../Figure/DegreeDistributionSardegna.png"
plt.savefig(path,dpi=300,bbox_inches='tight')
plt.show()