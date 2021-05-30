
import numpy as np
import functions as fn
import pandas as pd


data=pd.read_csv(r'C:\Users\Yash Priyadarshi\train.csv')
data = np.array(data)
m, n = data.shape

np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.


data_train = data[1000:10000].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

X_train=np.transpose( np.asmatrix(X_train))
Y_train=np.transpose( np.asmatrix(Y_train))
X_dev=np.transpose( np.asmatrix(X_dev))
Y_dev=np.transpose( np.asmatrix(Y_dev))

num_labels=10
input_layer_size=X_train.shape[1]
hidden_layer_size=20

maxiter = 100
lambda_reg = 1

initial_Theta1=fn.randinitialiseWeights(input_layer_size,hidden_layer_size)
initial_Theta2=fn.randinitialiseWeights(hidden_layer_size,10)
nn_params = np.concatenate(( np.array(initial_Theta1.flatten()), np.array(initial_Theta2.flatten()) ),axis=0)


myargs = ( X_train, Y_train, num_labels, lambda_reg, input_layer_size, hidden_layer_size)
results = op.minimize(fn.costFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

fn.get_accuracy(Theta1, Theta2, X_dev, Y_dev)

fn.predict(X_dev, Y_dev, Theta1, Theta2, 0)
fn.predict(X_dev, Y_dev, Theta1, Theta2, 1)
fn.predict(X_dev, Y_dev, Theta1, Theta2, 2)
fn.predict(X_dev, Y_dev, Theta1, Theta2, 3)
