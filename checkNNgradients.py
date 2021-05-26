import numpy as np
import functions as fn
import backpropagation as bp

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

            loss1,_ = fn.costFunction(nn_params1,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
            loss2,_ = fn.costFunction(nn_params2,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)

            numgrad1[i][j] = (loss2 - loss1) / (2*e)
            perturb1[i][j] = 0

    for i in range(Theta2.shape[0]):
        for j in range(Theta2.shape[1]):

            perturb2[i][j]= e

            nn_params1 = np.concatenate(( np.array((Theta1).flatten()), np.array((Theta2-perturb2).flatten()) ))
            nn_params2 = np.concatenate(( np.array((Theta1).flatten()), np.array((Theta2+perturb2).flatten()) ))


            loss1,_ = fn.costFunction(nn_params1,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
            loss2,_ = fn.costFunction(nn_params2,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
            numgrad2[i][j] = (loss2 - loss1) / (2*e)
            perturb2[i][j] = 0

    return numgrad1, numgrad2
    


def check_gradients(lammbda):

    input_layer_size = 10
    hidden_layer_size = 3
    num_labels = 10
    m = 10

    ##generating random test data 

    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    ##generating X using same function

    X_check = debugInitializeWeights(m, input_layer_size-1)
    #y  = 1 + np.transpose( np.mod(1:m, num_labels) )
    y_check = np.zeros((m,1))
    for i in range(m):
        y_check[i] = ((i+1)%num_labels) +1 

    nn_params = np.concatenate(( np.array(Theta1.flatten()), np.array(Theta2.flatten())), axis=0)

    cost = fn.costFunction(nn_params,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
    
    Xc_ones=np.insert(X_check, 0, 1, axis=1)

    grad1,grad2 = bp.backpropagation(Theta1,Theta2,Xc_ones,y_check,lammbda,num_labels)
    #numgrad = computeNumericalGradient(costFunc, nn_params);
    
    numgrad1,numgrad2 = computeNumericalGradient(Theta1,Theta2,X_check,y_check,lammbda,num_labels,input_layer_size,hidden_layer_size)

    grad = np.concatenate( ( np.array(grad1.flatten()), np.array(grad2.flatten()) ), axis=1)
    
    numgrad = nn_params = np.concatenate(( np.array(numgrad1.flatten()), np.array(numgrad2.flatten()) ), axis=0)
    # numgrad=[numgrad]
    # numgrad=np.array(numgrad)

    diff = np.linalg.norm(numgrad-grad)/ np.linalg.norm(numgrad+grad)

    print("if gradients are correct, diff should be less than 1e-9")
    print("diff: ",diff)

    return diff
