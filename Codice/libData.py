
# Library to handle [especially writing and reading] data

from pathlib import Path
mainFolder    = Path(__file__).resolve().parent
projectFolder = mainFolder.parent
dataFolder    = projectFolder/'Dati'

matrixZipPath = dataFolder/'MatriciPendolarismo1991.zip'
sizeZipPath   = dataFolder/'CensimentoRegioni1991.zip'
coordZipPath  = dataFolder/'LimitiRegioni1991.zip'
shpFilePath   = f"zip://{coordZipPath}!Limiti1991_g/Com1991_g/Com1991_g_WGS84.shp"

regDataZipPath = dataFolder/'DatiRegioni1991.zip'
simDataZipFile = dataFolder/'DatiSimulazione.zip'

import zipfile as zf
from zipfile import ZipFile as ZF

from io import TextIOWrapper as tiow
from io import BytesIO as bio
from io import StringIO as sio

import pandas as pd
import csv,json
import geopandas as gpd

import numpy as np

import libParameters as libP


### Main functions ###

def ExtractRegionData():
    dicMun,dicReg = ReadMunRegCodes()

    BuildAdjacencyMatrices(dicMun,dicReg)
    BuildSizeDistributions(dicReg)

    WriteRegionData(dicReg)

def LoadRegionData(code):
    el = {
        'A':'.txt',
        'W':'.txt',
        'li2Name':'.json',
        'li2Coord':'.txt',
        'name2li':'.json',
        'sizeDistr':'.txt',
        'Nc':'.json'
    }
    parameters = {}

    with ZF(regDataZipPath) as z:
        for data,ext in el.items():
            path = f'{code}/{data}{ext}'
            with z.open(path,'r') as f:
                match data:
                    case 'A' | 'W' | 'li2Coord' | 'sizeDistr':
                        parameters[data] = np.loadtxt(
                            tiow(f,encoding='utf-8'),
                            delimiter=",",dtype=int
                        )

                    case 'li2Name':
                        parameters[data] = {
                            int(k): v for k, v in json.load(f).items()
                        } # This is necessary since the keys of a «.json» file are always strings, hence they have to be converted to integer if originally they were such

                    case 'Nc' | 'name2li':
                        parameters[data] = json.load(f)

    return libP.Parameters(**parameters)


### Auxiliary functions ###

def xls2csv(
    xlsFile,
    zipFile=None
):
    if zipFile is None:
        b = xlsFile
    else:
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

def ReadMunRegCodes(): # Read Municipality-Region Codes
    # Conversion of «fileName» from the old format «.xls» to a more manageable «.csv»
    matrixCSV = xls2csv('elencom91.xls',matrixZipPath)

    # These two dictionary are necessary to link muicipalities and regions via their codes defined in «file», which will be useful later on to extract the actual data for the adjacency matrices
    dicMun = {} # Dictionary to link municipality codes with region codes
    dicReg = {
        i+1:{
            'li2Name':{},  # Dictionary to link local indices with the municipality name
            'li2Coord':{}, # Dictionary to link local indices with the municipality representative point
            'name2li':{},  # Dictionary to link local indices with the municipality name
            'Code2li':{},  # Dictionary to link municipality codes with local indices
            'Nc':0  # Number of cities in a region
        } for i in range(21)
    } # The index 21 is arbitrarily associated to Italy viewed as the 21th region, hence its local index is actually the global one

    gdf = gpd.read_file(shpFilePath).set_index('PRO_COM_T').geometry.representative_point()
    reader = csv.reader(matrixCSV)
    next(reader)  # Skip header line containing metadata labels
    for row in reader:
        try: # If the code is not empty
            codeReg = int(row[0]) # Region code
            codeMun = row[3]      # Municipality code
            nameMun = row[4]      # Municipality name

            dicMun[codeMun] = codeReg
            # In reality «codeMun» it's more like «(Province code)+(Municipality code)»

            UpdateDictionary(
                dicReg,
                codeReg,
                nameMun,
                codeMun,
                gdf
            )

            UpdateDictionary(
                dicReg,
                21,
                nameMun,
                codeMun,
                gdf
            )

        except ValueError:
            continue # Ignore it otherwise

    return dicMun, dicReg

def UpdateDictionary(
    dicReg,
    codeReg,
    nameMun,
    codeMun,
    gdf
):
    lgi = dicReg[codeReg]['Nc'] # Local/Global index
    dicReg[codeReg]['li2Name'][lgi] = nameMun
    dicReg[codeReg]['li2Coord'][lgi] = [gdf[codeMun].x,gdf[codeMun].y]
    dicReg[codeReg]['name2li'][nameMun] = lgi
    dicReg[codeReg]['Code2li'][codeMun] = lgi

    dicReg[codeReg]['Nc'] = lgi+1 # Update local/global number of cities
    # Local (codeReg=!=21) index
    # Global (codeReg==21) index

def BuildAdjacencyMatrices(
    dicMun,
    dicReg
):

    for r in dicReg:
        Nc = dicReg[r]['Nc']

        for (M,numTyp) in [
            ('A',np.uint8), # 'A' == [Unitary] Adjacency matrix
            ('W',np.int64)  # 'W' == Weighted adjacency matrix
        ]:
            dicReg[r][M] = np.zeros((Nc,Nc),dtype=numTyp)

        li2Coord = np.empty((Nc,2),np.float64)
        for i in range(Nc):
            li2Coord[i,0] = dicReg[r]['li2Coord'][i][0]
            li2Coord[i,1] = dicReg[r]['li2Coord'][i][1]
        dicReg[r]['li2Coord'] = li2Coord

    with ZF(matrixZipPath) as z, z.open('Pen_91It.txt') as f:
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
                            dicReg,
                            commuters,
                            oReg,dReg,
                            oMun,dMun
                        )

                    UpdateMatrices(
                        dicReg,
                        commuters,
                        21,21,
                        oMun,dMun
                    )
                except Exception:
                    continue

    # The if statement excludes municipalities whose codes are only partially written due to, I presume, typos from ISTAT
    # Instead, the try statement catches the few cases where the code does not match any municipality (so far only one: «022008»)

def UpdateMatrices(
    dicReg,commuters,
    oReg,dReg,
    oMun,dMun
):
    oI = dicReg[oReg]['Code2li'][oMun] # Local/Global origin index
    dI = dicReg[dReg]['Code2li'][dMun] # Local/Global destination index
    # The index is considered global iff «oReg=dReg=21»
    
    dicReg[oReg]['A'][oI,dI] = 1
    dicReg[dReg]['A'][dI,oI] = 1
    dicReg[oReg]['W'][oI,dI] += commuters
    dicReg[dReg]['W'][dI,oI] += commuters
    # The sum in «matricesReg[oReg]['W'][oI,dI] += commuters» is necessary as there are repeating origin-destination links in the dataset

def BuildSizeDistributions(
    dicReg
):
    sizeDistribution = {}

    with ZF(sizeZipPath) as z:
        Nc = dicReg[21]['Nc']
        sizeDistribution[21] = np.zeros((Nc,),dtype=np.int64)

        for r in range(1,21):
            Nc = dicReg[r]['Nc']
            sizeDistribution[r] = np.zeros((Nc,),dtype=np.int64)

            # Conversion of the relative «.xls» file into a more manageable «.csv» one
            sizeDistrFile = xls2csv(z.read(f'R{r:02d}_DatiCPA_1991.xls'))
            reader = csv.reader(sizeDistrFile)

            next(reader) # Skip header line containing metadata labels
            for row in reader:
                try: # If the code is not empty and it is a key
                    # In reality «codeMun» it's more like «(Province code)+(Municipality code)»
                    codeMun = f'{int(row[2]):006d}' # Municipality code
                    secPop  = int(row[5]) # Section population

                    li = dicReg[r]['Code2li'][codeMun]
                    sizeDistribution[r][li] += secPop

                    gi = dicReg[21]['Code2li'][codeMun]
                    sizeDistribution[21][gi] += secPop
                except ValueError:
                    continue # Ignore it otherwise

            dicReg[r]['sizeDistr'] = sizeDistribution[r]

        dicReg[21]['sizeDistr'] = sizeDistribution[21]

def WriteRegionData(
    dicReg
):
    with ZF(
        regDataZipPath,'w',
        compression=zf.ZIP_DEFLATED, # Enable compression
        compresslevel=9              # Max compression for «ZIP_DEFLATED»
        # https://docs.python.org/3/library/zipfile.html#zipfile-objects
    ) as z:
        el = {
            'A':'.txt',
            'W':'.txt',
            'li2Name':'.json',
            'li2Coord':'.txt',
            'name2li':'.json',
            'sizeDistr':'.txt',
            'Nc':'.json'
        }

        for r in range(21):
            folder = f'{'0' if r+1<10 else ''}{r+1}'

            for data,ext in el.items():
                path = f'{folder}/{data}{ext}'
                buf = sio()

                match ext:
                    case '.txt':
                        np.savetxt(
                            buf,
                            dicReg[r+1][data],
                            fmt="%d",
                            delimiter=","
                        )
                        
                    case '.json':
                        json.dump(
                            dicReg[r+1][data],
                            buf,
                            indent=3 if data != 'Nc' else 0
                        )

                z.writestr(path,buf.getvalue())

def WriteSimulationData(
    vrtState,
    snapshots,
    siVrtState,
    typ,
    lbl,
    li2Name,
    Ni,
    sid
):
    lbl = [t.replace('.','') for t in lbl]

    mode = 'a' if sid is not None and sid != 1 else 'w'
    with ZF(
        simDataZipFile,mode,
        compression=zf.ZIP_DEFLATED,
        compresslevel=9
    ) as z:
        for r in range(Ni):
            dicName2SortedPop = {
                t:{
                    li2Name[i]:vrtState[r,i,t] for i in siVrtState[r,::-1,t]
                } for t in typ
            }

            folder = (
                f'{'' if sid is None else f's{sid}/'}'
                f'{0 if r+1<10 else ""}{r+1}'
            )
            for t in typ:
                buf = bio()
                path = f'{folder}/{lbl[t]}CitySizesFinal.npy'
                np.save(buf,vrtState[r,:,t])
                z.writestr(path,buf.getvalue())

                buf = sio()
                path = f'{folder}/{lbl[t]}CitySizesSorted.json'
                json.dump(dicName2SortedPop[t],buf,indent=3)
                z.writestr(path,buf.getvalue())

                buf = bio()
                path = f'{folder}/{lbl[t]}Snapshots.npy'
                np.save(buf,snapshots[r,:,:,t])
                z.writestr(path,buf.getvalue())

def SetParameters(cls):
    parameters = libP.parameters
    for name,kwargs in parameters.items():
        setattr(cls,name,libP.Parameter(**kwargs))

def LoadCaseStudies(cls):
    data = libP.caseStudies

    caseStudies = data['list']
    selectedCS = data['selected']

    listCS = list(caseStudies.keys())
    dictCS = {}

    for name,study in caseStudies.items():
        dictCS[name] = {}

        for key,val in study.items():
            prm = getattr(cls,key)
            dictCS[name][prm] = val

        for (prmName,prmlist) in [
            ('region','regionList'),
            ('interactingLaw','intLawList'),
            ('studiedParameter','studiedPrmList')
        ]:
            prm = getattr(cls,prmName)
            value = dictCS[name][prm]
            dictCS[name][prm] = getattr(cls,prmlist).name[value]

    return dictCS, selectedCS, listCS
