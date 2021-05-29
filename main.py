# %%
from backpropagation import backpropagation
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import gradientDescent as gd
import scipy.optimize as op
import checkNNgradients as checkNN
# from mnist import MNIST
import scipy.io
#from keras.datasets import mnist
# from mlxtend.data import loadlocal_mnist
# %%
## function for loadung cifar 10 datatset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dataset=unpickle('data_batch_1')

# # %%
y=[dataset[b'labels']]
y=np.transpose(y)
y = np.asmatrix(y)

X=[dataset[b'data']]
X=np.transpose(X)
X = np.asmatrix(X)
X=np.transpose(X)


X_train=X[0:8000]
y_train=y[0:8000]

X_test=X[8001:10000]
y_test=y[8001:10000]
## loading X and y 

# (X_real, y_real), (X_test, y_test) = mnist.load_data()

# y=np.asmatrix(y_test)
# y=np.transpose(y)       

# X=np.zeros((784,10000))
# X=np.asmatrix(X)
# X=np.transpose(X)

# for i in range(10000):
#     X[i]=X_test[i].flatten()        ## X_test was (10000,28,28), so flattened the 28x28
    
##### loadingg done ###

# mat=scipy.io.loadmat('ex4data1.mat')
# X=mat['X']
# X=np.asmatrix(X)


# y=mat['y']
# y=np.asmatrix(y)

# for i in range(5000):
#     y[i]-=1

# mat=scipy.io.loadmat('ex4weights.mat')
# Theta1=mat['Theta1']
# Theta1=np.asmatrix(Theta1)
# Theta2=mat['Theta2']
# Theta2=np.asmatrix(Theta2)

# num_row,num_col=X.shape
# m=num_row #num of training examples
# lammbda=1
# input_layer_size=num_col
# hidden_layer_size=25
# num_labels=10

num_row,num_col=X_train.shape
m=num_row #num of training examples
lammbda=0.5
input_layer_size=num_col
hidden_layer_size=20
num_labels=10

# t=np.zeros((10,1))
# count=-5
# for i in range(10):
#     t[i]=count
#     count+=1

# t=fn.relu(t)


# %%

initial_Theta1=fn.randinitialiseWeights(num_col,hidden_layer_size)
initial_Theta2=fn.randinitialiseWeights(hidden_layer_size,10)


# initial_nn_params = np.concatenate(( np.array(initial_Theta1.flatten()), np.array(initial_Theta2.flatten()) ))
# initial_nn_params = np.concatenate(( np.array(Theta1.flatten()), np.array(Theta2.flatten()) ),axis=1)
nn_params = np.concatenate(( np.array(initial_Theta1.flatten()), np.array(initial_Theta2.flatten()) ),axis=0)

# %%

# these check variables were used to find if the backprop algo was working fine using the check_gradients function

# for this grad1, grad2 was given as output, now changing it for the optimization function



# nn_params =checkNN.check_gradients(lammbda)


# initial_nn_params=np.transpose(initial_nn_params)
# nn_params=gd.gradientDescentnn(X, y, initial_nn_params, 0.01, 100, 0.1, input_layer_size, hidden_layer_size, num_labels)


# initial_nn_params=np.transpose(np.asmatrix(initial_nn_params))

maxiter = 50
lambda_reg = 1

myargs = ( X_train, y_train, num_labels, lambda_reg, input_layer_size, hidden_layer_size)
results = op.minimize(fn.costFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

#get the solution of the optimization
# nn_params = res.x


# nn_params=gd.gradientDescentnn(X, y, initial_nn_params, 1,30, lammbda, input_layer_size, hidden_layer_size, num_labels)

# nn_params=np.transpose(nn_params)

Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

pred = fn.predict(Theta1, Theta2, X_test)

print('Training Set Accuracy: %f' % (np.mean(pred == y_test) * 100))

