
from multiprocessing import freeze_support

def main():
    import libGUIs
    import libData as libD
    import libNetworks as libN
    import libKTMAS as libK


    ### Graphic User Interface ###
    clsGUI = libGUIs.ParametersGUI()
    clsPrm = clsGUI.GatherParameters() # Parameters


    if clsPrm.simFlag:

        ### Data extraction ###
        if clsPrm.extraction: libD.ExtractRegionData()


        ### Matrices extraction ###
        clsReg = libD.LoadRegionData(clsPrm.region)


        ### Network analysis ###
        if clsPrm.analysis:
            clsNA = libN.NetworkAnalysis(clsPrm,clsReg)

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
        if clsPrm.parametricStudy:
            clsKS = libK.ParametricStudy(clsPrm,clsReg)
        else:
            clsKS = libK.KineticSimulation(clsPrm,clsReg)

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
