# %%
import numpy as np
import matplotlib.pyplot as plt
import functions

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
initial_Theta1=randinitialiseWeights(num_col+1,20)
initial_Theta2=randinitialiseWeights(21,10)

# %%

# %%
