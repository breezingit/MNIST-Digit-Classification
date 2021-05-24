# %%
import numpy as np
import functions

def backpropagation(initial_Theta1,initial_Theta2,X_ones,y,lammbda,num_labels):
    
    num_row,num_col=X.shape
    for i in range(num_row):
        a1=new_Col=np.transpose(X[0])
        z2 = np.dot(initial_Theta1,a1)

        a2=sigmoid(z2)

        new_Col=np.zeros((1, m))
        new_Col=np.transpose(new_Col)
        a2=np.insert(a2, 0, 1, axis=1)

        z3=np.dot(initial_Theta2,a2)

        hyp=sigmoid(z3)

        yT=zeros((num_labels,1))
        yT[y[i]]=1

        delta3=hyp-yT
        delta2=np.dot(np.transpose(initial_Theta2),delta3)
        delta2=np.multiply( delta2, sigmoidGradient(a2=np.insert(z2, 0, 1, axis=1)) ) 
        