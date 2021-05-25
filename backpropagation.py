# %%
import numpy as np
import functions as fn

def backpropagation(initial_Theta1,initial_Theta2,X_ones,y,lammbda,num_labels):
    
    num_row,num_col=X_ones.shape

    Theta1_grad = np.zeros(initial_Theta1.shape)
    Theta2_grad = np.zeros(initial_Theta2.shape)

    for i in range(num_row):
        a1=np.transpose(X_ones[0])
        z2 = np.dot(initial_Theta1,a1)

        a2=fn.sigmoid(z2)

        a2=np.insert(a2, 0, 1, axis=0)

        z3=np.dot(initial_Theta2,a2)

        hyp=fn.sigmoid(z3)

        yT= np.zeros((num_labels,1))
        yT[y[i-1]]=1

        delta3=hyp-yT
        delta2=np.dot(np.transpose(initial_Theta2),delta3)
        a2=np.insert(z2, 0, 1, axis=0)
        delta2=np.multiply( delta2, fn.sigmoidGradient(a2) ) 

        # tt1 = sig2(2:end)*a1';
        #             Theta1_grad = Theta1_grad + tt1;
        #             tt2 = sig3*a2';
        #             Theta2_grad = Theta2_grad + tt2;

        tt1 = np.dot(delta2[1:], np.transpose(a1)) 
        Theta1_grad = Theta1_grad + tt1

        tt2 = np.dot(delta3, np.transpose(a2))
        Theta2_grad = Theta2_grad + tt2

    Theta1_grad = Theta1_grad/num_row
    Theta2_grad = Theta2_grad/num_row


    #regularize

    add1 = lammbda/num_row*initial_Theta1
    add1[:0] = 0
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + add1[:,1:]
         
    add2 = lammbda/num_row*initial_Theta2
    add2[:,0] = 0
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + add2[:,1:]

    return Theta1_grad,Theta2_grad

        