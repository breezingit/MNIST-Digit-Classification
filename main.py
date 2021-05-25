# %%
from backpropagation import backpropagation
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import gradientcheck as gc

# %%
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dataset=unpickle('data_batch_1')

# %%
y=[dataset[b'labels']]
y=np.transpose(y)
y = np.asmatrix(y)

num_labels = 10
lammbda = 0

# %%
X=[dataset[b'data']]
X=np.transpose(X)
X = np.asmatrix(X)
X=np.transpose(X)
num_row,num_col=X.shape
m=num_row #num of training examples

# %%
X_ones=np.insert(X, 0, 1, axis=1)
# %%
##########  number of neurons in hidden layer==20
initial_Theta1=fn.randinitialiseWeights(num_col,20)
initial_Theta2=fn.randinitialiseWeights(20,10)

print(initial_Theta1.shape)
print(initial_Theta2.shape)


# %%

grad1,grad2 = backpropagation(initial_Theta1,initial_Theta2,X_ones,y,lammbda,num_labels)
gradchk = gc.computeGradientsCheck(X,y,initial_Theta1, initial_Theta2, num_labels, lammbda)

grad1
grad2
gradchk


# %%

# %%
