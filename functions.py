import numpy as np
counter=0

def randinitialiseWeights(L_in,L_out):
        
        epsilon_init = 44
        W =np.random.randint(-1*epsilon_init,epsilon_init,size = ( L_out,1+ L_in))
        W=np.divide(W, 1000)
     
        return W

def sigmoid(X):
        z = 1.0/(1.0 + np.exp(-X))
        return z

def sigmoidGradient(z):
        g=np.multiply( sigmoid(z), (np.ones(z.shape)-sigmoid(z)) )
        return g


def costFunction(nn_params, X, y ,num_labels,lammbda,input_layer_size,hidden_layer_size):
        global counter

        m=y.size

        initial_Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

        initial_Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

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
                yt[int(y[i].item())-1] = 1

                # temp = -1*(yt).*log(h) - (ones(num_labels,1) - (yt)).*log(ones(num_labels,1) - h);

                temp = -1*yt
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



        capdelta1 = np.zeros(initial_Theta1.shape)
        capdelta2 = np.zeros(initial_Theta2.shape)

        for i in range(m):
                

                flag1 = np.matrix(eX[i])
                a1=np.transpose(flag1)
                z2 = np.dot(initial_Theta1,a1)
                #z2 =np.dot(initial_Theta1, np.matrix(eXT[:,i]))

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
                capdelta1=capdelta1+ np.dot(delt2,np.matrix(eX[i,:]))

        Theta1_grad =np.multiply(capdelta1,1/m)
        Theta2_grad =np.multiply(capdelta2,1/m)

        Theta1_grad[:, 1:input_layer_size+1] = Theta1_grad[:, 1:input_layer_size+1] +np.multiply(initial_Theta1[:, 1:input_layer_size+1], (lammbda / m))
        Theta2_grad[:, 1:hidden_layer_size+1] = Theta2_grad[:, 1:hidden_layer_size+1] + np.multiply(initial_Theta2[:, 1:hidden_layer_size+1],(lammbda / m))

        grad = np.concatenate(( np.array(Theta1_grad.flatten()), np.array(Theta2_grad.flatten()) ), axis=1)
        
        counter=counter+1
        print(counter)
        return J,grad

def predict(initial_Theta1,initial_Theta2, X):
        m = X.shape[0]

        p = np.zeros(m)
        h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1),np.transpose(initial_Theta1)))
        h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), np.transpose(initial_Theta2)))
        p = np.argmax(h2, axis=1)
        return p