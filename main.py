import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dataset=unpickle('data_batch_1')

Y=[dataset[b'labels']]
Y=np.transpose(Y)
Y = np.asmatrix(Y)

X=[dataset[b'data']]
X=np.transpose(X)
X = np.asmatrix(X)
X=np.transpose(X)

m=len(X) #num of training examples

np.random.permutation(m)







