import numpy as np
import functions as fn
import backpropagation as bp

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

    return numgrad1, numgrad2
    


def check_gradients(X_check,y_check,Theta1,Theta2,lammbda, hidden_layer_size,input_layer_size,num_labels):

    ##generating X using same function

    
    
    #y  = 1 + np.transpose( np.mod(1:m, num_labels) )
    y_check = np.zeros((num_labels,1))
    for i in range(num_labels):
        y_check[i] = ((i+1)%num_labels) +1 

    nn_params = np.concatenate(( np.array(Theta1.flatten()), np.array(Theta2.flatten())), axis=0)

    #cost = fn.costFunction(nn_params,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
    
    #Xc_ones=np.insert(X_check, 0, 1, axis=1)

    _,grad1,grad2 = fn.costFunction(nn_params, X_check, y_check,10, 0, 20, 5)
    #numgrad = computeNumericalGradient(costFunc, nn_params);
    
    numgrad1,numgrad2 = computeNumericalGradient(Theta1,Theta2,X_check,y_check,lammbda,num_labels,input_layer_size,hidden_layer_size)

    grad = np.concatenate( ( np.array(grad1.flatten()), np.array(grad2.flatten()) ), axis=1)
    
    numgrad = nn_params = np.concatenate(( np.array(numgrad1.flatten()), np.array(numgrad2.flatten()) ), axis=0)
    # numgrad=[numgrad]
    # numgrad=np.array(numgrad)

    #diff = np.linalg.norm(numgrad-grad)/ np.linalg.norm(numgrad+grad)

    # print("if gradients are correct, diff should be less than 1e-9")
    # print("diff: ",diff)

    #return diff
    
    for i in range(len(numgrad)):
        print("Numerical Gradient = %f. BackProp Gradient = %f."%(grad[0][i],numgrad[i]))

