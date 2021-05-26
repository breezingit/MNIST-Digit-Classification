import numpy as np
import backpropagation as bp
import scipy
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

        return J

def predict(initial_Theta1,initial_Theta2, X):
        m = X.shape[0]

        p = np.zeros(m)
        h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1),np.transpose(initial_Theta1)))
        h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), np.transpose(initial_Theta2)))
        p = np.argmax(h2, axis=1)
        return p



def fminfunc(initial_nn_params,X,y,input_layer_size,hidden_layer_size,lammbda):

        num_col=X.shape[1]

        # costFunction = lambda p: fn.costFunction(p,X, y, num_labels, lammbda, num_col,20)

        result = scipy.optimize.fmin_cg(costFunction, x0=initial_nn_params, fprime=bp.backpropagation, \
                                args=(np.array(X.flatten()),y,lammbda),maxiter=50,disp=True,full_output=True)

        params=result[0]

        theta1 = params[:(input_layer_size+1)*hidden_layer_size] \
            .reshape((hidden_layer_size,input_layer_size+1))
        theta2 = params[(input_layer_size+1)*hidden_layer_size:] \
            .reshape((output_layer_size,hidden_layer_size+1))
    
        return theta1, theta2