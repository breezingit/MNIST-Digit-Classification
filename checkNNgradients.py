import numpy as np
import functions as fn
import backpropagation as bp
import sys

### creates a small neural network to check back propagation gradients ###

def debugInitializeWeights(fan_out, fan_in):

    w = np.zeros((fan_out, 1 + fan_in))
    # we initialise it with sin to ensure it remains same

    k = 1
    for i in range(fan_out):

        for j in range(fan_in+1):

            w[i, j] = np.sin(k)
            k = k+1
    return w


def computeNumericalGradient(Theta1,Theta2,X_check,y_check,lammbda,num_labels,input_layer_size,hidden_layer_size):
    numgrad1 = np.zeros((Theta1.shape))
    numgrad2 = np.zeros((Theta2.shape))
    perturb1 = np.zeros((Theta1.shape))
    perturb2 = np.zeros((Theta2.shape))
    e = 1e-4

    for i in range(Theta1.shape[0]):
        for j in range(Theta1.shape[1]):

            perturb1[i][j]= e
            #nn_params, X, y ,num_labels,lammbda,input_layer_size,hidden_layer_size

            nn_params1 = np.concatenate(( np.array((Theta1-perturb1).flatten()), np.array((Theta2).flatten()) ))
            nn_params2 = np.concatenate(( np.array((Theta1+perturb1).flatten()), np.array((Theta2).flatten()) ))

            loss1 = bp.backpropagation(nn_params1, X_check, y_check, lammbda, num_labels, hidden_layer_size, input_layer_size)
            loss2 = bp.backpropagation(nn_params2, X_check, y_check, lammbda, num_labels, hidden_layer_size, input_layer_size)

            numgrad1[i][j] = (loss2 - loss1) / (2*e)
            perturb1[i][j] = 0

    for i in range(Theta2.shape[0]):
        for j in range(Theta2.shape[1]):

            perturb2[i][j]= e

            nn_params1 = np.concatenate(( np.array((Theta1).flatten()), np.array((Theta2-perturb2).flatten()) ))
            nn_params2 = np.concatenate(( np.array((Theta1).flatten()), np.array((Theta2+perturb2).flatten()) ))


            loss1 = bp.backpropagation(nn_params1, X_check, y_check, lammbda, num_labels, hidden_layer_size, input_layer_size)
            loss2 = bp.backpropagation(nn_params2, X_check, y_check, lammbda, num_labels, hidden_layer_size, input_layer_size)
            numgrad2[i][j] = (loss2 - loss1) / (2*e)
            perturb2[i][j] = 0


    grad = np.concatenate(( np.array(numgrad1.flatten())  , np.array(numgrad2.flatten()) ), axis=0)
    return grad
    


def check_gradients(lammbda):

    input_layer_size=3
    hidden_layer_size=5
    num_labels=3
    m=5

    X_check=fn.randinitialiseWeights(check_input_size-1,check_hidden_size)

    y_check=np.zeros((check_m,1))

    for i in range(check_m):
        y_check[i]=i%check_numlabels



    thetacheck1=fn.randinitialiseWeights(check_input_size, check_hidden_size)
    thetacheck2=fn.randinitialiseWeights(check_hidden_size,check_numlabels)

    nn_params = np.concatenate(( np.array(Theta1.flatten()), np.array(Theta2.flatten())), axis=0)

    nn_params=np.asmatrix(nn_params)
    nn_params=np.transpose(nn_params)
    #cost = fn.costFunction(nn_params,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
    
    #Xc_ones=np.insert(X_check, 0, 1, axis=1)

    J,grad = fn.costFunction(nn_params, X_check, y_check, num_labels, lammbda, input_layer_size, hidden_layer_size)
    #numgrad = computeNumericalGradient(costFunc, nn_params);
    
    numgrad = computeNumericalGradient(Theta1,Theta2,X_check,y_check,lammbda,num_labels,input_layer_size,hidden_layer_size)

    # grad = np.concatenate( ( np.array(grad1.flatten()), np.array(grad2.flatten()) ), axis=1)
    
    # numgrad = nn_params = np.concatenate(( np.array(numgrad1.flatten()), np.array(numgrad2.flatten()) ), axis=0)
    # numgrad=[numgrad]
    # numgrad=np.array(numgrad)


    diff = np.linalg.norm(numgrad-grad)/ np.linalg.norm(numgrad+grad)

    #return diff
    
    for i in range(len(numgrad)):
        print("Numerical Gradient = %f. BackProp Gradient = %f."%(grad[0][i],numgrad[i]))

    print("if gradients are correct, diff should be less than 1e-9")
    print("diff: ",diff)

