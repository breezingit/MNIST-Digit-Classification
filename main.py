# %%
from backpropagation import backpropagation
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import gradientcheck as gc
import checkNNgradients as checkNN
import scipy.optimize as op
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
lammbda = 4

# %%
X=[dataset[b'data']]
X=np.transpose(X)
X = np.asmatrix(X)
X=np.transpose(X)
num_row,num_col=X.shape
m=num_row #num of training examples

input_layer_size=num_col
hidden_layer_size=20

# %%
X_ones=np.insert(X, 0, 1, axis=1)
# %%
##########  number of neurons in hidden layer==20
initial_Theta1=fn.randinitialiseWeights(num_col,20)
initial_Theta2=fn.randinitialiseWeights(20,10)

# %%5

#grad1,grad2 = backpropagation(initial_Theta1,initial_Theta2,X_ones,y,lammbda,num_labels)
#gradchk = gc.computeGradientsCheck(X,y,initial_Theta1, initial_Theta2, num_labels, lammbda)
# %%
#yo = checkNN.debugInitializeWeights(5,6)
#yo
#J = fn.costFunction(X,y,initial_Theta1,initial_Theta2,num_labels,lammbda)

#gradc=gc.computeGradientsCheck(X, y, initial_Theta1, initial_Theta2, num_labels, lammbda,20,3072,10)

# %%
# diff = checkNN.check_gradients(lammbda)
# diff

initial_nn_params = np.concatenate(( np.array(initial_Theta1.flatten()), np.array(initial_Theta2.flatten()) ))
# %%
#result = op.fmin_tnc(func = fn.costFunction, x0 = initial_Theta, fprime = gradient, args = (X,y))
# result[1]
# options= {'maxiter': 1}

costFunction = lambda p: fn.costFunction(p,X, y, num_labels, lammbda, num_col,20)

# res = op.minimize(costFunction, initial_nn_params, jac=True, method='TNC', options=options) 

# nn_params=res.x

Theta1,Theta2=fn.fminfunc(initial_nn_params, X, y, input_layer_size, hidden_layer_size,lammbda)


# %%
# Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],(hidden_layer_size, (input_layer_size + 1)))

# Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],(num_labels, (hidden_layer_size + 1)))



h1 = fn.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1),np.transpose(Theta1)))
h2 = fn.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), np.transpose(Theta2)))



pred = fn.predict(Theta1, Theta2, X)
# print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
# %%
