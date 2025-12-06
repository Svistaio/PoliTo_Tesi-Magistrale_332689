import zipfile as zf
from zipfile import ZipFile as ZF

from io import TextIOWrapper as tiow
from io import BytesIO as bio
from io import StringIO as sio

import pandas as pd
import csv

import numpy as np


### Main functions ###

def ExtractAdjacencyMatrices(zipFile):
    # Conversion of «elencom91.xls» from the old format .xls to a more modern .csv 
    with ZF(zipFile) as z:
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
    dicReg2Mun = {i:{'lI':0} for i in range(1,21)} # Local index
    dicMun2Reg = {}
    gI = 0 # Global index

    reader = csv.reader(streamCSV)
    next(reader)  # Skip header line containing metadata labels
    for row in reader:
        try: # If the code is not empty
            codeReg = int(row[0])

            codeMun = row[3]
            nameMun = row[4]
            list = [nameMun,gI,dicReg2Mun[codeReg]['lI']]
            dicReg2Mun[codeReg][codeMun] = list
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
    dicCode2Reg = {i:r for i,r in enumerate(regList,start=1)}
    
    matricesIt  = {
        'A':np.zeros((gI,gI),dtype=int), # 'A' == Adjacency matrix
        'W':np.zeros((gI,gI),dtype=int)  # 'W' == Weighted adjacency matrix
    }
    matricesReg = {
        k:{
            'A':np.zeros((dicReg2Mun[k]['lI'],dicReg2Mun[k]['lI']),dtype=int),
            'W':np.zeros((dicReg2Mun[k]['lI'],dicReg2Mun[k]['lI']),dtype=int)
        }
        for k in dicReg2Mun
    }

    with ZF(zipFile) as z, z.open("Pen_91It.txt") as f:
        for line in tiow(f,encoding="utf-8"):
            oMun = line[:6]         # Origin municipality
            dMun = line[11:17]      # Destination municipality
            commuters = int(line[17:-1]) # Edge weight (commuters)

            if oMun != dMun and ' ' not in dMun: # and ' ' not in oMun
                try:
                    oReg = dicMun2Reg[oMun] # Origin region
                    dReg = dicMun2Reg[dMun] # Destination region

                    if oReg == dReg: # and commuters!=0
                        oI = dicReg2Mun[oReg][oMun][2] # [Local] origin index
                        dI = dicReg2Mun[dReg][dMun][2] # [Local] destination index
                        
                        matricesReg[oReg]['A'][oI,dI] = 1
                        matricesReg[dReg]['A'][dI,oI] = 1
                        matricesReg[oReg]['W'][oI,dI] += commuters
                        matricesReg[dReg]['W'][dI,oI] = matricesReg[oReg]['W'][oI,dI]
                        # The sum in «matricesReg[oReg]['W'][oI,dI] += commuters» is necessary as there are repeating origin-destination links in the dataset

                    oI = dicReg2Mun[oReg][oMun][1] # [Global] origin index
                    dI = dicReg2Mun[dReg][dMun][1] # [Global] destination index
                        
                    matricesIt['A'][oI,dI] = 1
                    matricesIt['A'][dI,oI] = 1
                    matricesIt['W'][oI,dI] += commuters
                    matricesIt['W'][dI,oI] = matricesIt[1,oI,dI]
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
        for i,M in enumerate(matricesReg.values(),start=1):
            path = (f"{"0" if i<9 else ""}{i}"
                    f"AdjacencyMatrix{dicCode2Reg[i]}.txt")
            Save2Zip(M['A'],path,z)

            path = (f"{"0" if i<9 else ""}{i}"
                    f"WeightedAdjacencyMatrix{dicCode2Reg[i]}.txt")
            Save2Zip(M['W'],path,z)

        Save2Zip(matricesIt['A'],"AdjacencyMatrixIt.txt",z)
        Save2Zip(matricesIt['W'],"WeightedAdjacencyMatrixIt.txt",z)

def ReadAdjacencyMatrices(zipFile,idA,idW):
    with ZF(zipFile) as z:
        A = MatrixFromZip(z,idA)
        W = MatrixFromZip(z,idW)
    return A, W


### Auxiliary functions ###

def Save2Zip(M,path,z):
    buf = sio()
    np.savetxt(buf,M,fmt="%d",delimiter=",")
    data = buf.getvalue()
    z.writestr(path,data)

def MatrixFromZip(z,id):
    with z.open(id,"r") as f:
        M = np.loadtxt( # Decodes binary stream as UTF-8 text
            tiow(f,encoding="utf-8"),
            delimiter=",",
            dtype=int
        )
    return M