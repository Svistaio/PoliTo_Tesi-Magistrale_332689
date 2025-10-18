from zipfile import ZipFile as zf

from io import TextIOWrapper as tiow
from io import BytesIO as bio
from io import StringIO as sio
import pandas as pd

import csv

import numpy as np


zipPath = "../Dati/matrici_pendolarismo_1991.zip"

# List entries of the zip file «matrici_pendolarismo_1991.zip»
with zf(zipPath) as z:
    print(z.namelist())


# Conversion of «elencom91.xls» from the old format .xls to a more modern .csv 
with zf(zipPath) as z:
    b = z.read("elencom91.xls")  # Extract raw bytes from «elencom91.xls»


streamXLS = bio(b)  # Creates a binary stream in RAM from the bytes «b»
df = pd.read_excel( # Reads the Excel content
    streamXLS,      # from «streamXLS»
    dtype=str,      # treatig columns as strings
    engine="xlrd",  # using «xlrd»
    sheet_name=0    # for only the first worksheet
) # «df» stands for «DataFrame» from «pandas.DataFrame»
streamCSV = sio()   # Allocates a text buffer in RAM that behaves like a writable text file (UTF-8 encoding)
df.to_csv(          # Serializes «df» as CSV
    streamCSV,      # into «streamCSV» and
    index=False     # without the DataFrame’s row numbers
)
streamCSV.seek(0)   # Resets the text buffer cursor to the start so it can be read from the beginning


# Dictionary and number of the major municipalities
dicMun = {}
m = 0

reader = csv.reader(streamCSV)
next(reader)  # skip header line containing metadata labels
for row in reader:
    try: # If the code is not empty
        codeR = int(row[0])
        if codeR == 20: # If it matches Sardinia region code
            key   = row[3]
            value = [row[4], m]
            dicMun[key] = value
            m+=1
    except ValueError:
        continue # Ignore it otherwise


A = np.zeros((m,m))

with zf(zipPath) as z:
    with z.open("Pen_91It.txt") as f:
        for line in tiow(f,encoding="utf-8"):
            oM = line[:6]    # Origin municipality
            dM = line[11:17] # Destination municipality
            if oM != dM:
                if oM in dicMun and dM in dicMun:
                    oI = dicMun[oM] # Origin index
                    dI = dicMun[dM] # Destination index
                    A[oI,dI] = 1
                    A[dI,oI] = 1

            

