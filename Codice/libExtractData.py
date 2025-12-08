
import zipfile as zf
from zipfile import ZipFile as ZF

from io import TextIOWrapper as tiow
from io import BytesIO as bio
from io import StringIO as sio

import pandas as pd
import csv

import numpy as np


### Main functions ###

def ExtractAdjacencyMatrices():
    zipFile = '../Dati/MatriciPendolarismo1991.zip'

    # Conversion of «elencom91.xls» from the old format «.xls» to a more modern «.csv»
    csvFile = xls2csv(zipFile,'elencom91.xls')

    dicMun, dicReg = ReadMunRegCodes(csvFile)

    BuildAdjacencyMatrices(
        dicMun,dicReg,
        zipFile,'Pen_91It.txt'
    )

    WriteAdjacencyMatrices(
        '../Dati/DatiPendolarismo1991.zip',dicReg
    )

def ReadAdjacencyMatrices(code):
    zipFile = '../Dati/DatiPendolarismo1991.zip'
    pathA = f'{code}/A.txt'
    pathW = f'{code}/W.txt'

    with ZF(zipFile) as z:
        A = MatrixFromZip(z,pathA)
        W = MatrixFromZip(z,pathW)
    return A, W


### Auxiliary functions ###

def xls2csv(zipFile,xlsFile):
    with ZF(zipFile) as z:
        b = z.read(xlsFile)  # Extract raw bytes from «elencom91.xls»

    streamXls = bio(b)  # Create a binary stream in RAM from the bytes «b»
    df = pd.read_excel( # Read the Excel content
        streamXls,      # from «streamXLS»
        dtype=str,      # treating columns as strings
        engine="xlrd",  # using «xlrd»
        sheet_name=0    # for only the first worksheet
    ) # «df» stands for «DataFrame» from «pandas.DataFrame»
    streamCsv = sio()   # Allocate a text buffer in RAM that behaves like a writable text file (UTF-8 encoding)
    df.to_csv(          # Serialize «df» as «.csv»
        streamCsv,      # into «streamCSV» and
        index=False     # without the DataFrame’s row numbers
    )
    streamCsv.seek(0)   # Reset the text buffer cursor to the start so it can be read from the beginning

    return streamCsv

def ReadMunRegCodes(file):
    # These two dictionary are necessary to link muicipalities and regions via their codes defined in «file», which will be useful later on to extract the actual data for the adjacency matrices
    dicMun = {} # Dictionary to link municipality codes with region codes
    dicReg = {
        i+1:{
            'li2Name':{}, # Dictionary to link local indices with the municipality name
            'Code2li':{}, # Dictionary to link municipality code with local indices
            'Nc':0  # Number of cities in a region
        } for i in range(21)
    } # The index 21 is arbitrarily associated to Italy viewed as the 21th region, hence its local index is actually the global one

    reader = csv.reader(file)
    next(reader)  # Skip header line containing metadata labels
    for row in reader:
        try: # If the code is not empty
            codeReg = int(row[0]) # Region code
            codeMun = row[3]      # Municipality code
            nameMun = row[4]      # Municipality name

            dicMun[codeMun] = codeReg

            li = dicReg[codeReg]['Nc'] # Local index
            dicReg[codeReg]['li2Name'][li] = nameMun
            dicReg[codeReg]['Code2li'][codeMun] = li

            gi = dicReg[21]['Nc']      # Global index
            dicReg[21]['li2Name'][gi] = nameMun
            dicReg[21]['Code2li'][codeMun] = gi

            dicReg[codeReg]['Nc'] += 1 # Update local number of cities
            dicReg[21]['Nc'] +=1       # Upadte global number of cities

            # In reality «codeMun» it's more like «Province code + Municipality code»

        except ValueError:
            continue # Ignore it otherwise

    return dicMun, dicReg

def BuildAdjacencyMatrices(
    dicMun,dicReg,
    zipFile,txtFile
):
    for r in dicReg:
        for M in ['A','W']:
            dicReg[r][M] = np.zeros(
                (dicReg[r]['Nc'],dicReg[r]['Nc']),dtype=int
            )
            # 'A' == [Unitary] Adjacency matrix
            # 'W' == Weighted adjacency matrix

    with ZF(zipFile) as z, z.open(txtFile) as f:
        for line in tiow(f,encoding="utf-8"):
            oMun = line[:6]         # Origin municipality
            dMun = line[11:17]      # Destination municipality
            commuters = int(line[17:-1]) # Edge weight (commuters)

            if oMun != dMun and ' ' not in dMun: # and ' ' not in oMun
                try:
                    oReg = dicMun[oMun] # Origin region
                    dReg = dicMun[dMun] # Destination region

                    if oReg == dReg: # and commuters!=0
                        oI = dicReg[oReg]['Code2li'][oMun] # Local origin index
                        dI = dicReg[dReg]['Code2li'][dMun] # Local destination index
                        
                        dicReg[oReg]['A'][oI,dI] = 1
                        dicReg[dReg]['A'][dI,oI] = 1
                        dicReg[oReg]['W'][oI,dI] += commuters
                        dicReg[dReg]['W'][dI,oI] = dicReg[oReg]['W'][oI,dI]
                        # The sum in «matricesReg[oReg]['W'][oI,dI] += commuters» is necessary as there are repeating origin-destination links in the dataset

                    oI = dicReg[21]['Code2li'][oMun] # Global origin index
                    dI = dicReg[21]['Code2li'][dMun] # Global destination index
                        
                    dicReg[21]['A'][oI,dI] = 1
                    dicReg[21]['A'][dI,oI] = 1
                    dicReg[21]['W'][oI,dI] += commuters
                    dicReg[21]['W'][dI,oI] = dicReg[1,oI,dI]
                except Exception:
                    continue

    # The if statement excludes municipalities whose codes are only partially written due to, I presume, typos from ISTAT
    # Instead, the try statement catches the few cases where the code does not match any municipality (so far only one: «022008»)

def WriteAdjacencyMatrices(zipPath,dicReg):
    with ZF(
        zipPath,'w',
        compression=zf.ZIP_DEFLATED, # Enable compression
        compresslevel=9              # Max compression for «ZIP_DEFLATED»
        # https://docs.python.org/3/library/zipfile.html#zipfile-objects
    ) as z:
        for r in range(21):
            folder = f'{'0' if r+1<=9 else ''}{r+1}'

            for M in ['A', 'W']:
                path = f'{folder}/{M}.txt'
                Save2Zip(dicReg[r+1][M],path,z)

def Save2Zip(M,path,z):
    buf = sio()
    np.savetxt(
        buf,M,
        fmt="%d",
        delimiter=","
    )
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
