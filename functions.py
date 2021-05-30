import numpy as np
import matplotlib.pyplot as plt
import neuralNetwork as nn

counter=0

def randinitialiseWeights(L_in,L_out):
        
        # epsilon_init = 44
        # W =np.random.randint(-1*epsilon_init,epsilon_init,size = ( L_out,1+ L_in))
        # W=np.divide(W, 1000)
     
        # return W
        epi = (6**1/2) / (L_in + L_out)**1/2
    
        W = np.random.rand(L_out,L_in+1) *(2*epi) -epi
        # W = np.random.rand(L_out,L_in) *(2*epi) 
        
        return W

def relu(Z):
    return np.maximum(Z, 0)

def reluGradient(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def costFunction(nn_params, X, y ,num_labels,lammbda,input_layer_size,hidden_layer_size):
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
                
                # col=np.matrix(eXT[:,i])
                # col=np.transpose(col)
                
                z2 = np.dot(initial_Theta1,eXT[:,i])
                # z2 = np.dot(initial_Theta1,col)

                # a2=expit(z2)

                a2=relu(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                # hyp=expit(z3)
                hyp=softmax(z3)
                
                hyp[hyp==0]=10**-15
                hyp[hyp==1]=1-10**-15
                
                # for j in range(hyp.shape[0]):
                #         hyp[j]=0.001
                # for t in range(hyp.shape[0]):
                #         if(hyp[t]<=0):
                #                 hyp[t]=0.001
                
                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                #yt[int(y[i].item())-1] = 1

                # yt[int(y[i])-1]=1

                yt[int(y[i])]=1  # for matlab indexing

                # temp = -1*(yt).*log(h) - (ones(num_labels,1) - (yt)).*log(ones(num_labels,1) - h);

                temp = -1*yt

                
                temp = np.multiply(temp,np.log(hyp))
                # temp = np.multiply(temp,np.log(np.array(hyp).astype(float)))

                
                
                
                one_minus_hyp=np.ones(hyp.shape)-hyp

                log_one_hyp=np.log(one_minus_hyp)
                
                temp = temp - np.multiply((np.ones((num_labels,1)) - yt), log_one_hyp)

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

        eX= np.insert(X, 0, 1, axis=1)
        eXT=np.transpose(eX)

        for i in range(m):
                
                # col=np.matrix(eXT[:,i])
                # col=np.transpose(col)
                
                z2 = np.dot(initial_Theta1,eXT[:,i])
                # z2 = np.dot(initial_Theta1,col)

                # a2=expit(z2)
                a2=relu(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                # hyp=expit(z3)
                hyp=softmax(z3)

                hyp[hyp==0]=10**-15
                hyp[hyp==1]=1-10**-15

                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                # yt[int(y[i])-1] = 1
                yt[int(y[i])] = 1

                delt3=hyp-yt

                # delt3=np.transpose( np.asmatrix([hyp]))-yt
                delt2=  np.dot(np.transpose(initial_Theta2[:,1:]) ,delt3)


                # z2=np.asmatrix([sigmoidGradient(z2)])

                # delt2=np.multiply(delt2,sigmoidGradient(z2))
                delt2=np.multiply(delt2,reluGradient(z2))
                # a2=np.asmatrix([a2])  # using as matrix gave out transpose(a2)
                capdelta2=capdelta2+ np.dot(delt3,np.transpose(a2))

                # capdelta2=capdelta2+ np.dot(delt3,np.transpose(a2))
                
                # var=np.asmatrix([eXT[:,i]])
                capdelta1=capdelta1+ np.dot(delt2,np.transpose(eXT[:,i]))
                # capdelta1=capdelta1+ np.dot(delt2,var)

        Theta1_grad =np.multiply(capdelta1,1/m)
        Theta2_grad =np.multiply(capdelta2,1/m)

        Theta1_grad[:, 1:input_layer_size+1] = Theta1_grad[:, 1:input_layer_size+1] +np.multiply(initial_Theta1[:, 1:input_layer_size+1], (lammbda / m))
        Theta2_grad[:, 1:hidden_layer_size+1] = Theta2_grad[:, 1:hidden_layer_size+1] + np.multiply(initial_Theta2[:, 1:hidden_layer_size+1],(lammbda / m))

        # Theta1_grad[:, 1:input_layer_size+1] = Theta1_grad[:, 1:input_layer_size+1] +initial_Theta1[:, 1:input_layer_size+1]*(lammbda / m)
        # Theta2_grad[:, 1:hidden_layer_size+1] = Theta2_grad[:, 1:hidden_layer_size+1] +initial_Theta2[:, 1:hidden_layer_size+1]*(lammbda / m)

        grad = np.concatenate(( np.array(Theta1_grad.flatten())  , np.array(Theta2_grad.flatten()) ), axis=1)
        
        
        # return J,Theta1_grad,Theta2_grad
        
        counter+=1
        print(J,counter)
        return J,grad


def get_accuracy(initial_Theta1,initial_Theta2, X,y):
        m = X.shape[0]
        print(X.shape)
        p = np.zeros(m)
        h1 = relu(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1),np.transpose(initial_Theta1)))
        h2 = softmax(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), np.transpose(initial_Theta2)))
        
        
        p = np.argmax(h2, axis=1)
        print('Training Set Accuracy: %f' % (np.mean(p == y) * 100))
        

def predict(X,Y,Theta1,Theta2,index):
        test = X[index,:, None]
        m=X.shape[0]
        h1 = relu(np.dot(np.concatenate([np.ones((m,1)), X], axis=1),np.transpose(Theta1)))
        h2 = softmax(np.dot(np.concatenate([np.ones((m,1)), h1], axis=1), np.transpose(Theta2)))
        predictions = np.argmax(h2,1)
        
        print("Prediction: ", predictions[index][0])
        
        test = test.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(test, interpolation='nearest')
        plt.show()

