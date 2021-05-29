import numpy as np
import functions as fn
import backpropagation as bp

counter=0



def gradientDescentnn(X,y,initial_nn_params,alpha,num_iters,lammbda,input_layer_size, hidden_layer_size, num_labels):
    global counter

    Theta1 = initial_nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = initial_nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    
    m=len(y)
    #J_history =[]
    
    for i in range(num_iters):
        nn_params = np.append(np.array( Theta1.flatten()),np.array( Theta2.flatten()))
        J,grad1, grad2 = fn.costFunction(nn_params, X, y, num_labels, lammbda, input_layer_size, hidden_layer_size)
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        alpha=1/(100+i*50)
        print(counter+1,J)
        
        counter=counter+1
        
    
    nn_params = np.concatenate((np.array( Theta1.flatten()),np.array( Theta2.flatten())),axis=1)
    return nn_params