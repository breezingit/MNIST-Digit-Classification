import numpy as np
import functions as fn

def sigmoid(X):
        z = 1.0/(1.0 + np.exp(-X))
        return z

def sigmoidGradient(z):
        g=np.multiply( sigmoid(z), (np.ones(z.shape)-sigmoid(z)) )
        return g


def backpropagation(nn_params,X,y,lammbda,num_labels,hidden_layer_size,input_layer_size):
    
        global counter

        m=y.size

        # initial_Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
        #                 (hidden_layer_size, (input_layer_size + 1)))

        # initial_Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
        #                 (num_labels, (hidden_layer_size + 1)))


        initial_Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
        initial_Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

        J = 0

        eX= np.insert(X, 0, 1, axis=1)
        eXT=np.transpose(eX)

        for i in range(m):
                #a1=np.transpose(np.matrix(X[i]))

                z2 = np.dot(initial_Theta1,eXT[:,i])

                a2=sigmoid(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                hyp=sigmoid(z3)

                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                yt[int(y[i].item())] = 1

                # temp = -1*(yt).*log(h) - (ones(num_labels,1) - (yt)).*log(ones(num_labels,1) - h);

                temp = -1*yt

                hyp=np.asmatrix([hyp])
                hyp=np.transpose(hyp)
                temp = np.multiply(temp,np.log(hyp))

                temp = temp - np.multiply((np.ones((num_labels,1)) - yt), np.log(np.ones((num_labels,1))- hyp))

                J = J + np.sum(temp)
        

        J = J/m

        # reg = sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));
        #  reg = reg*lambda/m;
        #  reg = reg/2;
         
        #  J = J + reg;

        reg = np.sum( np.multiply(initial_Theta1[:,1:], initial_Theta1[:,1:])) + np.sum( np.multiply(initial_Theta2[:,1:] , initial_Theta2[:,1:]))

        reg = reg*lammbda/m
        reg = reg/2

        J = J + reg

        return J