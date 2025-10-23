import zipfile as zf
from zipfile import ZipFile as ZF

from io import TextIOWrapper as tiow
from io import BytesIO as bio
from io import StringIO as sio
import pandas as pd

import csv

import numpy as np


zipPath = "../Dati/matriciPendolarismo1991.zip"

# List entries of the zip file
# with ZF(zipPath) as z:
#     print(z.namelist())


# Conversion of «elencom91.xls» from the old format .xls to a more modern .csv 
with ZF(zipPath) as z:
    b = z.read("elencom91.xls")  # Extract raw bytes from «elencom91.xls»


streamXLS = bio(b)  # Create a binary stream in RAM from the bytes «b»
df = pd.read_excel( # Read the Excel content
    streamXLS,      # from «streamXLS»
    dtype=str,      # treating columns as strings
    engine="xlrd",  # using «xlrd»
    sheet_name=0    # for only the first worksheet
) # «df» stands for «DataFrame» from «pandas.DataFrame»
streamCSV = sio()   # Allocate a text buffer in RAM that behaves like a writable text file (UTF-8 encoding)
df.to_csv(          # Serialize «df» as CSV
    streamCSV,      # into «streamCSV» and
    index=False     # without the DataFrame’s row numbers
)
streamCSV.seek(0)   # Reset the text buffer cursor to the start so it can be read from the beginning


# Dictionary and number of the major municipalities
# dicMun = {}
dicReg2Mun = {i: {'lI':0} for i in range(1,21)}
dicMun2Reg = {}
gI = 0

reader = csv.reader(streamCSV)
next(reader)  # Skip header line containing metadata labels
for row in reader:
    try: # If the code is not empty
        codeReg = int(row[0])

        codeMun = row[3]
        nameMun = row[4]
        value = [nameMun,gI,dicReg2Mun[codeReg]['lI']]
        dicReg2Mun[codeReg][codeMun] = value
        dicMun2Reg[codeMun] = codeReg

        dicReg2Mun[codeReg]['lI'] += 1
        gI+=1
    except ValueError:
        continue # Ignore it otherwise

regList = [
    'Piemonte',
    'ValledAosta',
    'Lombardia',
    'Trentino-AltoAdige',
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
    'Sardegna'
]
dicReg = {i:r for i,r in enumerate(regList,start=1)}

adjacencyMIt  = np.zeros((gI,gI))
adjacencyMReg = {
    k:np.zeros((dicReg2Mun[k]['lI'],dicReg2Mun[k]['lI']),dtype=int)
    for k in dicReg2Mun
}

with ZF(zipPath) as z, z.open("Pen_91It.txt") as f:
    for line in tiow(f,encoding="utf-8"):
        oMun = line[:6]    # Origin municipality
        dMun = line[11:17] # Destination municipality
        if oMun != dMun and ' ' not in dMun: # and ' ' not in oMun
            try:
                oReg = dicMun2Reg[oMun] # Origin region
                dReg = dicMun2Reg[dMun] # Destination region
                if oReg == dReg:
                    oI = dicReg2Mun[oReg][oMun][2] # [Local] origin index
                    dI = dicReg2Mun[dReg][dMun][2] # [Local] destination index
                    adjacencyMReg[oReg][oI,dI] = 1
                    adjacencyMReg[dReg][dI,oI] = 1
            except Exception:
                continue

# The if statement excludes municipalities whose codes are only partially written due to, I presume, typos from ISTAT
# Instead, the try statement catches the few cases where the code does not match any municipality (so far only one: «022008»)

with ZF(
    "../Dati/AdjacencyMatricesIt91.zip",
    "w",
    compression=zf.ZIP_DEFLATED, # Enable compression
    compresslevel=9              # Max compression for «ZIP_DEFLATED»
    # https://docs.python.org/3/library/zipfile.html#zipfile-objects
) as z:
    for i,A in enumerate(adjacencyMReg.values()):
        buf = sio()
        np.savetxt(buf,A,fmt="%d",delimiter=",")
        data = buf.getvalue()

        path = ("0" if i<9 else "")+ \
               str(i+1)+ \
               "AdjacencyMatrix"+ \
               regList[i]+ \
               ".txt"
        z.writestr(path,data)