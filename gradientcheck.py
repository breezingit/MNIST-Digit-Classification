import numpy as np
import functions as fn
import backpropagation as bp

#Helper Functions for interacting with other classes:
def getParams(w1,w2):
    #Get Wl and W2 unrolled into vector:
    params = np.concatenate(( np.array(w1.flatten()), np.array(w2.flatten()) ))
    return params

def setParams(params, hiddenLayerSize, inputLayerSize,outputLayerSize):
    #Set wi and W2 using single paramater vector.
    W1_start = 0
    W1_end = hiddenLayerSize * inputLayerSize
    W1 = np. reshape (params[W1_start:W1_end], (inputLayerSize , hiddenLayerSize))
    W2_end = W1_end + hiddenLayerSize*outputLayerSize
    W2 = np.reshape(params[W1_end: W2_end], (hiddenLayerSize, outputLayerSize))
    return W1,W2
# def computeGradients( X, y):
#     dJdw1, dJdw2 = costFunctionPrime(X, y)
#     return np.concatenate(dJdw1.ravel(), dJdw2.ravel())

##########################

def computeGradientsCheck(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda,hiddenLayerSize,inputLayerSize,outputLayerSize):
    paramsInitial = getParams(initial_Theta1,initial_Theta2)
    chkgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4
    hls=hiddenLayerSize
    ils=inputLayerSize
    ols=outputLayerSize

    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        
        initial_Theta1,initial_Theta2= setParams(paramsInitial + perturb,hls,ils,ols)
        
        loss2 = fn.costFunction(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda)

        initial_Theta1,initial_Theta2= setParams(paramsInitial - perturb,hls,ils,ols)
        lossl = fn.costFunction(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda)

        #Compute Check Gradient
        chkgrad[p] = (loss2 - lossl) / (2*e)
        #Return the value we changed to zero:
        perturb[p] = 0
    
    #Return Params to original value
    setParams(paramsInitial)

    return chkgrad


