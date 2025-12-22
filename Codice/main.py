
from multiprocessing import freeze_support


def main():
    import libGUI
    import libExtractData as libED
    import libAnalyseNetwork as libAN
    import libMASKineticTheory as libKT

    import importlib
    importlib.reload(libGUI)
    importlib.reload(libED)
    importlib.reload(libAN)
    importlib.reload(libKT)


    ### Graphic User Interface ###
    clsGUI = libGUI.ParametersGUI()
    clsPrm = clsGUI.GatherParameters() # Parameters


    if clsPrm.simFlag:

        ### Data extraction ###
        if clsPrm.extraction: libED.ExtractAdjacencyMatrices()


        ### Matrices extraction ###
        clsReg = libED.ReadAdjacencyMatrices(clsPrm.region)


        ### Network analysis ###
        if clsPrm.analysis:
            clsNA = libAN.NetworkAnalysis(clsPrm,clsReg)

            clsNA.DegreeDistributionFig()
            clsNA.WeightDistributionFig()
            clsNA.StrengthDistributionFig()

            clsNA.BetweennessCentralityFig()
            clsNA.StrengthVsDegreeFig()

            clsNA.AClusteringCoefficientFig()
            clsNA.WClusteringCoefficientFig()

            clsNA.AAssortativityFig()
            clsNA.WAssortativityFig()

            # clsNA.ShowFig()


        ### Kinetic simulation ###
        clsKS = libKT.KineticSimulation(clsPrm,clsReg)
        clsKS.MonteCarloSimulation()

        clsKS.SizeDistrFittingsFig()
        clsKS.AverageSizeFig()
        clsKS.SizeVsDegreeFig()
        clsKS.SizeDistrEvolutionFig()
        clsKS.SizeEvolutionsFig()

        # clsKS.ShowFig()

if __name__ == "__main__":
    freeze_support()
    main()
