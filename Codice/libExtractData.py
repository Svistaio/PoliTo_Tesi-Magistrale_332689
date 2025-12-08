
import zipfile as zf
from zipfile import ZipFile as ZF

from io import TextIOWrapper as tiow
from io import BytesIO as bio
from io import StringIO as sio

import pandas as pd
import csv, json

import numpy as np


### Main functions ###

def ExtractAdjacencyMatrices():
    zipFile = '../Dati/MatriciPendolarismo1991.zip'

    # Conversion of «elencom91.xls» from the old format «.xls» to a more manageable «.csv»
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
    dicReg = {}

    el = {
        'A':'.txt','W':'.txt',
        'li2Name':'.json',
        'Name2li':'.json',
        'Nc':'.json'
    }

    with ZF(zipFile) as z:
        for data in el:
            path = f'{code}/{data}{el[data]}'
            with z.open(path,'r') as f:
                match data:
                    case 'A' | 'W':
                        dicReg[data] = np.loadtxt(
                            tiow(f,encoding='utf-8'),
                            delimiter=",",
                            dtype=int
                        ) # Decodes binary stream as UTF-8 text

                    case 'li2Name':
                        dicReg[data] = {
                            int(k): v for k, v in json.load(f).items()
                        } # This is necessary since the keys of a «.json» file are always strings, hence they have to be converted to integer if originally they were such

                    case 'Nc' | 'Name2li':
                        dicReg[data] = json.load(f)

    return dicReg


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
            'Name2li':{}, # Dictionary to link local indices with the municipality name
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
            dicReg[codeReg]['Name2li'][nameMun] = li
            dicReg[codeReg]['Code2li'][codeMun] = li

            gi = dicReg[21]['Nc']      # Global index
            dicReg[21]['li2Name'][gi] = nameMun
            dicReg[21]['Name2li'][nameMun] = gi
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
                        UpdateMatrices(
                            dicReg,commuters,
                            oReg,dReg,oMun,dMun
                        )

                    UpdateMatrices(
                        dicReg,commuters,
                        21,21,oMun,dMun
                    )
                except Exception:
                    continue

    # The if statement excludes municipalities whose codes are only partially written due to, I presume, typos from ISTAT
    # Instead, the try statement catches the few cases where the code does not match any municipality (so far only one: «022008»)

def UpdateMatrices(
    dicReg,commuters,
    oReg,dReg,oMun,dMun
):
    oI = dicReg[oReg]['Code2li'][oMun] # Local/Global origin index
    dI = dicReg[dReg]['Code2li'][dMun] # Local/Global destination index
    # The index is considered global iff «oReg=dReg=21»
    
    dicReg[oReg]['A'][oI,dI] = 1
    dicReg[dReg]['A'][dI,oI] = 1
    dicReg[oReg]['W'][oI,dI] += commuters
    dicReg[dReg]['W'][dI,oI] += commuters
    # The sum in «matricesReg[oReg]['W'][oI,dI] += commuters» is necessary as there are repeating origin-destination links in the dataset

def WriteAdjacencyMatrices(zipPath,dicReg):
    with ZF(
        zipPath,'w',
        compression=zf.ZIP_DEFLATED, # Enable compression
        compresslevel=9              # Max compression for «ZIP_DEFLATED»
        # https://docs.python.org/3/library/zipfile.html#zipfile-objects
    ) as z:
        el = {
            'A':'.txt','W':'.txt',
            'li2Name':'.json',
            'Name2li':'.json',
            'Nc':'.json'
        }

        for r in range(21):
            folder = f'{'0' if r+1<=9 else ''}{r+1}'

            for data in el:
                buf = sio()
                path = f'{folder}/{data}{el[data]}'

                match data:
                    case 'A' | 'W':
                        np.savetxt(
                            buf,dicReg[r+1][data],
                            fmt="%d",delimiter=","
                        ) # Save the matrix in the zip as a «.txt» file
                        
                    case 'Name2li':
                        json.dump(
                            dicReg[r+1][data],
                            buf,indent=3
                        ) # Save the dictionary in the zip as a «.json» file

                    case 'li2Name':
                        json.dump(
                            dicReg[r+1][data],
                            buf,indent=3
                        ) # Save the dictionary in the zip as a «.json» file

                    case 'Nc':
                        json.dump(dicReg[r+1][data],buf)
                        # Save the number of cities in the zip as a «.json» file

                value = buf.getvalue()
                z.writestr(path,value)
