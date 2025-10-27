import numpy as np

global networkState


### Main code ###

# def Initialize(pop):

# def Initialize(MonteCarlo):


### Auxiliary code ###

class networkState:
    def __init__(self,totP,Nn,A,Nt):
        # Uniform initial state for all vertices
        self.verticesState   = np.ones((Nn,1),dtype=float)*(totP/Nn)
        self.averageState    = np.zeros((Nt,1),dtype=float)
        self.averageState[0] = totP/Nn

        self.Nn              = Nn # Number of nodes
        self.A               = A  # Adjacency matrix

    def updateState(self,dt):
        oldState = self.verticesState
        newState = np.copy(self.verticesState)
        P = np.random.permutation(self.Nn)

        halfN = np.floor(self.Nn/2)
        p1 = P[:halfN]; p2 = P[halfN+1:]

        for i in range():
            theta = np.random.binomial(1,self.A[p1(i)]*self.A[p2(i)]*dt)
            # newState(p1(i)) = oldState(p1(i))(1-theta)+
            # newState(p2(i)) = oldState(p2(i))(1-theta)+

        self.verticesState = newState
