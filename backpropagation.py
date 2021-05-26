
import numpy as np
import functions as fn

def backpropagation(nn_params,X,y,lammbda,num_labels):
    
        global counter

        m=y.size

        initial_Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

        initial_Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

        capdelta1 = np.zeros(initial_Theta1.shape)
        capdelta2 = np.zeros(initial_Theta2.shape)

        eX= np.insert(X, 0, 1, axis=1)
        eXT=np.transpose(eX)

        for i in range(m):
                
                z2 =np.dot(initial_Theta1, np.asmatrix(eXT[:,i]))

                a2=sigmoid(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                hyp=sigmoid(z3)

                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                yt[int(y[i].item())-1] = 1

                delt3=hyp-yt
                delt2=  np.dot(np.transpose(initial_Theta2[:,1:]) ,delt3)

                delt2=np.multiply(delt2,sigmoidGradient(z2))

                capdelta2=capdelta2+ np.dot(delt3,np.transpose(a2))
                capdelta1=capdelta1+ np.dot(delt2,np.transpose(eXT[:,i]))

        Theta1_grad =np.multiply(capdelta1,1/m)
        Theta2_grad =np.multiply(capdelta2,1/m)

        Theta1_grad[:, 1:input_layer_size+1] = Theta1_grad[:, 1:input_layer_size+1] +np.multiply(initial_Theta1[:, 1:input_layer_size+1], (lammbda / m))
        Theta2_grad[:, 1:hidden_layer_size+1] = Theta2_grad[:, 1:hidden_layer_size+1] + np.multiply(initial_Theta2[:, 1:hidden_layer_size+1],(lammbda / m))

        grad = np.concatenate(( x, np.array(Theta2_grad.flatten()) ), axis=1)
        
        counter=counter+1
        print(counter)
        return grad

        