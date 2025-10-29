import numpy as np
from tqdm import tqdm


### Main code ###

def MonteCarlo(totP,Nn,A,Nt,dt,l,a,sigma):
    stateCities = networkState(totP,Nn,A,Nt,l,a,sigma)

    for nt in tqdm(range(Nt),desc="Updating states"):
        stateCities.updateState(dt,nt)

    return stateCities 


### Auxiliary code ###

class networkState:
    def __init__(self,totP,Nn,A,Nt,l,a,sigma):
        # Uniform initial state for all vertices
        self.verticesState   = np.ones((Nn,1),dtype=float)*(totP/Nn)
        self.averageState    = np.zeros((Nt+1,1),dtype=float)
        self.averageState[0] = totP/Nn

        self.Nn              = Nn # Number of nodes
        self.A               = A  # Adjacency matrix

        def E(si,sr): return NonLinearEmigration(si,sr,l,a)
        self.E = E

        def mu(E): return StochasticFluctuations(sigma,E)
        self.mu = mu

    def updateState(self,dt,nt):
        oldState = self.verticesState
        newState = np.copy(self.verticesState)
        P = np.random.permutation(self.Nn)

        halfN = int(np.floor(self.Nn/2))
        p1 = P[:halfN]; p2 = P[halfN+1:]

        for i in range(halfN):
            p = float(np.clip(self.A[p1[i],p2[i]]*dt,0,1))
            theta = np.random.binomial(1,p)
            si = oldState[p1[i]]; sr = oldState[p2[i]]
            # It's assumed node p1(i) is the interacting node while node p2(i) is the receiving one

            E = self.E(si,sr); mu = self.mu(E)
            newState[p1[i]] = si*(1-theta)+theta*si*(1-E+mu)
            newState[p2[i]] = sr*(1-theta)+theta*(sr+si*E)

        self.verticesState = newState
        self.averageState[nt+1] = np.mean(newState)


def NonLinearEmigration(
        si, # Interacting city size
        sr, # Receiving city size
        l,  # Maximum emigration rate
        a   # Emigration intensity
    ):
    if si != 0:
        rs = sr/si           # Relative population
        ef = l*(rs**a)/(1+rs**a) # Actual emigration rate
    else:
        ef = 0
    return ef


def StochasticFluctuations(sigma,E):
    while True:
        mu = np.random.normal(0,sigma,size=1)
        if mu>E-1 and mu<E:
            break
    # The conditions «mu>E-1» and «mu<E» are necessary to have the total emigration rage 1-E+μ between 0 and 1
    return mu