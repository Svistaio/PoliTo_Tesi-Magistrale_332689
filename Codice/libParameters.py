
# Library to more easily define/change parameters

from dataclasses import dataclass
from typing import Any

from copy import deepcopy
import numpy as np

from pathlib import Path


### Module attributes ###

mainFolder     = Path(__file__).resolve().parent
projectFolder  = mainFolder.parent
dataFolder     = projectFolder/'Dati'

matrixZipPath  = dataFolder/'MatriciPendolarismo1991.zip'
sizeZipPath    = dataFolder/'CensimentoRegioni1991.zip'
coordZipPath   = dataFolder/'LimitiRegioni1991.zip'
shpFilePath    = f"zip://{coordZipPath}!Limiti1991_g/Com1991_g/Com1991_g_WGS84.shp"

regDataZipPath = dataFolder/'DatiRegioni1991.zip'
simDataZipFile = dataFolder/'DatiSimulazione.zip'

parameters = {
    "population": {
        "text": "S",
        "val": 1
    },
    "attractivity": {
        "text": "λ",
        "val": 1.0
    },
    "convincibility": {
        "text": "α",
        "val": 1.0
    },
    "deviation": {
        "text": "σ",
        "val": 1.0
    },
    "region": {
        "text": "Region selected",
        "val": "region"
    },
    "zetaFraction": {
        "text": "ζ",
        "val": 1.0
    },
    "timestep": {
        "text": "Δt",
        "val": 1.0
    },
    "timesteps": {
        "text": "Nt",
        "val": 1
    },
    "iterations": {
        "text": "Ni",
        "val": 1
    },
    "progressBar": {
        "text": "Progress Bar",
        "val": True
    },
    "extraction": {
        "text": "Extract data",
        "val": False
    },
    "analysis": {
        "text": "Network analysis",
        "val": False
    },
    "edgeWeights": {
        "text": "Edge weights",
        "val": False
    },
    "fluctuations": {
        "text": "Fluctuations",
        "val": True
    },
    "interactingLaw": {
        "text": "Interacting law",
        "val": "law"
    },
    "PdfPopUp": {
        "text": "Open PDF",
        "val": False
    },
    "LaTeXConversion": {
        "text": "LaTeX Conversion",
        "val": False
    },
    "snapshots": {
        "text": "Ns",
        "val": 100
    },
    "smoothingFactor": {
        "text": "Sf",
        "val": 10
    },
    "parametricStudy": {
        "text": "Parametric study",
        "val": False
    },
    "studiedParameter": {
        "text": "Studied parameter",
        "val": "study"
    },
    "startValuePrmStudy": {
        "text": "Start",
        "val": 1.0
    },
    "endValuePrmStudy": {
        "text": "End",
        "val": 1.0
    },
    "numberPrmStudy": {
        "text": "Nv",
        "val": 1
    }
}

caseStudies = {
    "selected": "λ(rsk^α)/(1+rsk^α)",
    "list": {
        "Default": {
            "attractivity": 0.05,
            "convincibility": 0.5,
            "deviation": 0.05,
            "region": 19,
            "zetaFraction": 0.1,
            "timestep": 0.01,
            "timesteps": int(1e5),
            "iterations": 3,
            "progressBar": False,
            "extraction": False,
            "analysis": False,
            "edgeWeights": False,
            "fluctuations": True,
            "interactingLaw": 3,
            "PdfPopUp": False,
            "LaTeXConversion": False,
            "snapshots": 100,
            "smoothingFactor": 10,
            "studiedParameter": 1,
            "startValuePrmStudy": 0.1,
            "endValuePrmStudy": 1.0,
            "numberPrmStudy": 3,
            "parametricStudy": False
        },
        "λ(rsk/α)/(1+rsk/α)": {
            "attractivity": 0.05,
            "deviation": 0.05,
            "region": 19,
            "zetaFraction": 0.1,
            "timestep": 0.01,
            "timesteps": int(5e5),
            "iterations": 15,
            "progressBar": False,
            "extraction": False,
            "analysis": False,
            "edgeWeights": False,
            "fluctuations": True,
            "interactingLaw": 1,
            "PdfPopUp": False,
            "LaTeXConversion": False,
            "snapshots": 100,
            "smoothingFactor": 10,
            "studiedParameter": 0,
            "startValuePrmStudy": 1.0,
            "endValuePrmStudy": 1.0,
            "numberPrmStudy": 1,
            "parametricStudy": False
        },
        "(1-ζ)efl_k/α+ζefs_k": {
            "attractivity": 0.05,
            "deviation": 0.05,
            "region": 19,
            "zetaFraction": 0.1,
            "timestep": 0.01,
            "timesteps": int(1e7),
            "iterations": 15,
            "progressBar": False,
            "extraction": False,
            "analysis": False,
            "edgeWeights": False,
            "fluctuations": True,
            "interactingLaw": 2,
            "PdfPopUp": False,
            "LaTeXConversion": False,
            "snapshots": 100,
            "smoothingFactor": 10,
            "studiedParameter": 0,
            "startValuePrmStudy": 0.01,
            "endValuePrmStudy": 0.1,
            "numberPrmStudy": 5,
            "parametricStudy": True
        },
        "λ(rsk^α)/(1+rsk^α)": {
            "attractivity": 0.01,
            "convincibility": 0.4,
            "deviation": 0.05,
            "region": 19,
            "zetaFraction": 0.1,
            "timestep": 0.01,
            "timesteps": int(1e7),
            "iterations": 15,
            "progressBar": False,
            "extraction": False,
            "analysis": False,
            "edgeWeights": True,
            "fluctuations": False,
            "interactingLaw": 3,
            "PdfPopUp": False,
            "LaTeXConversion": False,
            "snapshots": 100,
            "smoothingFactor": 10,
            "studiedParameter": 1,
            "startValuePrmStudy": 0.3,
            "endValuePrmStudy": 0.5,
            "numberPrmStudy": 5,
            "parametricStudy": False
        },
        "λ[rsk/(1+rsk)]^α": {
            "attractivity": 0.05,
            "convincibility": 0.3,
            "deviation": 0.05,
            "region": 19,
            "zetaFraction": 0.1,
            "timestep": 0.01,
            "timesteps": int(1e7),
            "iterations": 15,
            "progressBar": False,
            "extraction": False,
            "analysis": False,
            "edgeWeights": False,
            "fluctuations": True,
            "interactingLaw": 5,
            "PdfPopUp": False,
            "LaTeXConversion": False,
            "snapshots": 100,
            "smoothingFactor": 10,
            "studiedParameter": 1,
            "startValuePrmStudy": 0.3,
            "endValuePrmStudy": 1,
            "numberPrmStudy": 3,
            "parametricStudy": False
        }
    }
}

regionList = [
    'Piemonte',
    "Valle d'Aosta",
    'Lombardia',
    'Trentino-Alto Adige',
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
    'Sardegna',
    'Italia'
]

intLawList = [
    'λ(rs^α)/(1+rs^α)',         # 0
    'λ(rsk/α)/(1+rsk/α)',       # 1
    '(1-ζ)efl_k/α+ζefs_k/α',    # 2
    'λ(rsk^α)/(1+rsk^α)',       # 3
    '(1-ζ)efl_k^α+ζefs_k^α',    # 4
    'λ[rsk/(1+rsk)]^α',         # 5
    '(1-ζ)[efl_k]^α+ζ[efs_k]^α' # 6
]

prmStudyList = [
    parameters['attractivity']['text'],
    parameters['convincibility']['text'],
    parameters['zetaFraction']['text']
]

regPopDict = { # Italian region sizes in 1991
    'Piemonte':int(4302565),
    "Valle d'Aosta":int(115938),
    'Lombardia':int(8856074),
    'Trentino-Alto Adige':int(890360),
    'Veneto':int(4380797),
    'Friuli-Venezia Giulia':int(1197666),
    'Liguria':int(1676282),
    'Emilia-Romagna':int(3909512),
    'Toscana':int(3529946),
    'Umbria':int(811831),
    'Marche':int(1429205),
    'Lazio':int(5140371),
    'Abruzzo':int(1249054),
    'Molise':int(330900),
    'Campania':int(5630280),
    'Puglia':int(4031885),
    'Basilicata':int(610528),
    'Calabria':int(2070203),
    'Sicilia':int(4966386),
    'Sardegna':int(1648248),
    'Italia':int(56778031)
} # See Table 6.1 on p. 488 of «ISTAT Popolazione e abitazioni 1991 {04-12-2025}.pdf»

workersShMTemplate = {
    'handles':[],
    'parameters':{
        'Mdt': None,
        'wOdt': None,
        'wI': None,
        'di': None,
        'idi': None,
        'ns': None
    },
    'gui':{
        'progress': np.int64,
        'elapsed': np.float64,
        'done': np.int8
    }
}


### Main classes and function ###

@dataclass(eq=False)
class Parameter:
    text: Any = None
    val: Any = None
    var: Any = None
    lbl: Any = None
    wid: Any = None
    frame: Any = None
    lst: Any = None
    cbid: Any = None

class Parameters():
    def __init__(self,**kwargs):
        for text,value in kwargs.items():
            setattr(self,text,value)

class ComboBoxList():
    def __init__(self,attribute,lst):
        attribute.lst = lst
        self.name = lst
        self.code = {
            r:i for i,r in enumerate(lst)
        }

def CopyWorkerShMTemplate(): return deepcopy(workersShMTemplate)


### Discarded code ###

#region Alternative implementation to store all the module attributes but inside a «.json» file; it was discarded, even though it's quite neat, to justify the existance of «libParameters.py», which without those attributes would only contain three very short classes
""" File structure of «paramters.json»
    {
        "parameters":{
            "population": {
                "text":"S",
                "val":1
            },
            "attractivity": {
                "text":"λ",
                "val":1.0
            },
            "convincibility": {
                "text":"α",
                "val":1.0
            },
            "deviation": {
                "text":"σ",
                "val":1.0
            },
            "region": {
                "text":"Region selected",
                "val":"region"
            },
            "zetaFraction": {
                "text":"ζ",
                "val":1.0
            },
            "timestep": {
                "text":"Δt",
                "val":1.0
            },
            "timesteps": {
                "text":"Nt",
                "val":1
            },
            "iterations": {
                "text":"Ni",
                "val":1
            },
            "progressBar": {
                "text":"Progress Bar",
                "val":true
            },
            "extraction": {
                "text":"Extract data",
                "val":false
            },
            "analysis": {
                "text":"Network analysis",
                "val":false
            },
            "edgeWeights": {
                "text":"Edge weights",
                "val":false
            },
            "interactingLaw": {
                "text":"Interacting law",
                "val":"law"
            },
            "PdfPopUp": {
                "text":"Open PDF",
                "val":false
            },
            "LaTeXConversion": {
                "text":"LaTeX Conversion",
                "val":false
            },
            "snapshots": {
                "text":"Ns",
                "val":100
            },
            "smoothingFactor": {
                "text":"Sf",
                "val":10
            },
            "parametricStudy": {
                "text":"Parametric study",
                "val":false
            },
            "studiedParameter": {
                "text":"Studied parameter",
                "val":"study"
            },
            "startValuePrmStudy": {
                "text":"Start",
                "val":1.0
            },
            "endValuePrmStudy": {
                "text":"End",
                "val":1.0
            },
            "numberPrmStudy": {
                "text":"Nv",
                "val":1
            }
        },
        "caseStudies": {
            "selected": "Default",
            "list":{
                "Default": {
                    "attractivity": 0.05,
                    "convincibility": 0.5,
                    "deviation": 0.05,
                    "region": 19,
                    "zetaFraction": 0.1,
                    "timestep": 0.01,
                    "timesteps": 100000,
                    "iterations": 3,
                    "progressBar": false,
                    "extraction": false,
                    "analysis": false,
                    "edgeWeights": false,
                    "interactingLaw": 3,
                    "PdfPopUp": false,
                    "LaTeXConversion": false,
                    "studiedParameter": 1,
                    "snapshots": 100,
                    "smoothingFactor": 10,
                    "parametricStudy": false,
                    "startValuePrmStudy": 0.1,
                    "endValuePrmStudy": 1.0,
                    "numberPrmStudy": 3
                },
                "λ(rsk/α)/(1+rsk/α)": {
                    "attractivity": 0.05,
                    "deviation": 0.05,
                    "region": 19,
                    "timestep": 0.01,
                    "timesteps": 500000,
                    "iterations": 15,
                    "progressBar": false,
                    "extraction": false,
                    "analysis": false,
                    "edgeWeights": false,
                    "interactingLaw": 1,
                    "PdfPopUp": false,
                    "LaTeXConversion": false,
                    "snapshots": 100,
                    "smoothingFactor": 10,
                    "studiedParameter": 0,
                    "parametricStudy": false,
                    "startValuePrmStudy": 1.0,
                    "endValuePrmStudy": 1.0,
                    "numberPrmStudy": 1
                },
                "(1-ζ)efl_k/α+ζefs_k/": {
                    "attractivity": 0.05,
                    "deviation": 0.05,
                    "region": 19,
                    "timestep": 0.01,
                    "timesteps": 10000000,
                    "iterations": 15,
                    "progressBar": false,
                    "extraction": false,
                    "analysis": false,
                    "edgeWeights": false,
                    "interactingLaw": 2,
                    "PdfPopUp": false,
                    "LaTeXConversion": false,
                    "studiedParameter": 0,
                    "snapshots": 100,
                    "smoothingFactor": 10,
                    "parametricStudy": false,
                    "startValuePrmStudy": 1.0,
                    "endValuePrmStudy": 1.0,
                    "numberPrmStudy": 1
                },
                "λ(rsk^α)/(1+rsk^α)": {
                    "attractivity": 0.05,
                    "convincibility": 0.3,
                    "deviation": 0.05,
                    "region": 19,
                    "zetaFraction": 0.1,
                    "timestep": 0.01,
                    "timesteps": 10000000,
                    "iterations": 15,
                    "progressBar": false,
                    "extraction": false,
                    "analysis": false,
                    "edgeWeights": false,
                    "interactingLaw": 3,
                    "PdfPopUp": false,
                    "LaTeXConversion": false,
                    "studiedParameter": 0,
                    "snapshots": 100,
                    "smoothingFactor": 10,
                    "parametricStudy": false,
                    "startValuePrmStudy": 1.0,
                    "endValuePrmStudy": 1.0,
                    "numberPrmStudy": 1
                },
                "λ[rsk/(1+rsk)]^α": {
                    "attractivity": 0.05,
                    "convincibility": 0.3,
                    "deviation": 0.05,
                    "region": 19,
                    "zetaFraction": 0.1,
                    "timestep": 0.01,
                    "timesteps": 10000000,
                    "iterations": 15,
                    "progressBar": false,
                    "extraction": false,
                    "analysis": false,
                    "edgeWeights": false,
                    "interactingLaw": 5,
                    "PdfPopUp": false,
                    "LaTeXConversion": false,
                    "studiedParameter": 0,
                    "snapshots": 100,
                    "smoothingFactor": 10,
                    "parametricStudy": false,
                    "startValuePrmStudy": 1.0,
                    "endValuePrmStudy": 1.0,
                    "numberPrmStudy": 1
                }
            }
        }
    }
"""
""" Functions in «libData.py»
    def SetParameters(cls):
        caseStudiesPath = mainFolder/'parameters.json'
        text = caseStudiesPath.read_text(encoding="utf-8")
        data = json.loads(text)

        parameters = data['parameters']
        for name,kwargs in parameters.items():
            setattr(cls,name,libP.Parameter(**kwargs))

    def LoadCaseStudies(cls):
        caseStudiesPath = mainFolder/'parameters.json'
        text = caseStudiesPath.read_text(encoding="utf-8")
        data = json.loads(text)

        caseStudies = data['caseStudies']['list']
        selectedCS = data['caseStudies']['selected']
        listCS = list(caseStudies.keys())
        dictCS = {}

        for name,study in caseStudies.items():
            dictCS[name] = {}

            for key,val in study.items():
                prm = getattr(cls,key)
                dictCS[name][prm] = val

            for (prmName,prmlist) in [
                ('region','regList'),
                ('interactingLaw','intLawList'),
                ('studiedParameter','studiedPrmList')
            ]:
                prm = getattr(cls,prmName)
                value = dictCS[name][prm]
                dictCS[name][prm] = getattr(cls,prmlist).Name[value]

        return dictCS, selectedCS, listCS
"""
#endregion Of course, even the functions in «libData.py» have to chage as shown
