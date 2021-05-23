# %%
import numpy as np
import matplotlib.pyplot as plt
import randInitialise
# %%
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dataset=unpickle('data_batch_1')

# %%
Y=[dataset[b'labels']]
Y=np.transpose(Y)
Y = np.asmatrix(Y)

# %%
num_row,num_col=X.shape
X=[dataset[b'data']]
X=np.transpose(X)
X = np.asmatrix(X)
X=np.transpose(X)
m=num_row #num of training examples

# %%
new_Col=np.zeros((1, m))
new_Col=np.transpose(new_Col)
# %%
X_ones=np.insert(X, 0, 1, axis=1)
# %%
##########  number of neurons in hidden layer==20
initial_Theta1=randInitialiseWeights()
# %%
