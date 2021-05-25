import numpy as np

def randinitialiseWeights(L_in,L_out):
        
        epsilon_init = 1200
        W =np.random.randint(epsilon_init,size = ( L_out,1+ L_in))
        W=W / 10000
        l,r= W.shape
        print(l,r)
        #print(r)
        return W

def sigmoid(X):
        z = 1/(1 + np.exp(-X))
        return z

def sigmoidGradient(z):
        g=np.multiply( sigmoid(z), (np.ones(z.shape)-sigmoid(z)) )
        return g


def costFunction(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda):

        num_row,num_col=X.shape

        J = 0

        for i in range(num_row):
                a1=np.transpose(X[i-1])
                a1 = np.insert(a1, 0, 1, axis=1)

                z2 = np.dot(initial_Theta1,a1)

                a2=sigmoid(z2)

                a2=np.insert(a2, 0, 1, axis=1)

                z3=np.dot(initial_Theta2,a2)

                hyp=sigmoid(z3)

                yt = np.zeros(num_labels,1)
                yt[y[i-1]] = 1

                # temp = -1*(yt).*log(h) - (ones(num_labels,1) - (yt)).*log(ones(num_labels,1) - h);

                temp = -1*yt
                temp = np.multiply(temp,np.log(hyp))

                temp = temp - np.multiply((np.ones(num_labels,1) - yt), np.log(np.ones(num_labels,1)- hyp))

                J = J + np.sum(temp)
        

        J = J/num_row

        # reg = sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));
        #  reg = reg*lambda/m;
        #  reg = reg/2;
         
        #  J = J + reg;

        reg = np.sum( np.multiply(initial_Theta1[:,1:], initial_Theta1[:,1:])) + np.sum( np.multiply(initial_Theta2[:,1:] , initial_Theta2[:,1:]))

        reg = reg*lammbda/num_row
        reg = reg/2

        J = J + reg

        return J