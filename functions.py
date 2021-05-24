import numpy as np

def randinitialiseWeights(L_in,L_out):
        
        epsilon_init = 10;
        W =np.random.randint(epsilon_init, size=(1+L_in,L_out))
        W=W * 2 * epsilon_init - epsilon_init
        return W

def sigmoid(X):
        z = 1/(1 + np.exp(-X))
        return z

def sigmoidGradient(z):
        g=np.multiply( sigmoid(z), (np.ones(z.shape)-sigmoid(z)) )
        return g