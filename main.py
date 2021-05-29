# %%
from backpropagation import backpropagation
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import gradientDescent as gd
import scipy.optimize as op
# from mnist import MNIST

from keras.datasets import mnist
# from mlxtend.data import loadlocal_mnist
# %%
## function for loadung cifar 10 datatset

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


# dataset=unpickle('data_batch_1')

# %%
# y=[dataset[b'labels']]
# y=np.transpose(y)
# y = np.asmatrix(y)


# %%
# X=[dataset[b'data']]
# X=np.transpose(X)
# X = np.asmatrix(X)
# X=np.transpose(X)


## loading X and y 

(X_real, y_real), (X_test, y_test) = mnist.load_data()

y=np.asmatrix(y_test)
y=np.transpose(y)       

X=np.zeros((784,10000))
X=np.asmatrix(X)
X=np.transpose(X)

for i in range(10000):
    X[i]=X_test[i].flatten()        ## X_test was (10000,28,28), so flattened the 28x28
    
##### loadingg done ###

num_row,num_col=X.shape
m=num_row #num of training examples
lammbda=1
input_layer_size=num_col
hidden_layer_size=15
num_labels=10

# y=np.zeros((m,1))


# %%

initial_Theta1=fn.randinitialiseWeights(num_col,hidden_layer_size)
initial_Theta2=fn.randinitialiseWeights(hidden_layer_size,10)


initial_nn_params = np.concatenate(( np.array(initial_Theta1.flatten()), np.array(initial_Theta2.flatten()) ))

# %%

# these check variables were used to find if the backprop algo was working fine using the check_gradients function

# for this grad1, grad2 was given as output, now changing it for the optimization function

# X_check=fn.randinitialiseWeights(19, 50)
# y_check=fn.randinitialiseWeights(0, 50)

# thetacheck1=fn.randinitialiseWeights(20, 5)
# thetacheck2=fn.randinitialiseWeights(5, 10)

# nn_params =checkNN.check_gradients(X_check, y_check, thetacheck1, thetacheck2, 0, 5,20, 10)



# nn_params=gd.gradientDescentnn(X, y, initial_nn_params, 0.8, 10, 0, input_layer_size, hidden_layer_size, num_labels)


# nn_params=np.transpose(np.asmatrix(nn_params))

# costFunction = lambda p: fn.costFunction(p, X, y, num_labels, lammbda, input_layer_size, hidden_layer_size)

# options= {'maxiter': 100}

# res = op.minimize(costFunction,
#                         initial_nn_params,
#                         jac=True,
#                         method='TNC',
#                         options=options)

# #get the solution of the optimization
# nn_params = res.x


nn_params=gd.gradientDescentnn(X, y, initial_nn_params, 0.8,30, lammbda, input_layer_size, hidden_layer_size, num_labels)

nn_params=np.transpose(nn_params)

Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

pred = fn.predict(Theta1, Theta2, X)

print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
