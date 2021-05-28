import numpy as np
import scipy.optimize as op

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dataset=unpickle('data_batch_1')

y=[dataset[b'labels']]
y=np.transpose(y)
y = np.asmatrix(y)

num_labels = 10
lammbda = 4

X=[dataset[b'data']]
X=np.transpose(X)
X = np.asmatrix(X)
X=np.transpose(X)
num_row,num_col=X.shape
m=num_row #num of training examples

input_layer_size=num_col
hidden_layer_size=20

def getParams(w1,w2):
    #Get Wl and W2 unrolled into vector:
    params = np.concatenate(( np.array(w1.flatten()), np.array(w2.flatten()) ),axis=0)
    return params

def setParams(params, hiddenLayerSize, inputLayerSize,outputLayerSize):
    #Set wi and W2 using single paramater vector.
    W1_start = 0
    W1_end = hiddenLayerSize * inputLayerSize
    W1 = np. reshape (params[W1_start:W1_end], (inputLayerSize , hiddenLayerSize))
    W2_end = W1_end + hiddenLayerSize*outputLayerSize
    W2 = np.reshape(params[W1_end: W2_end], (hiddenLayerSize, outputLayerSize))
    return W1,W2

def randinitialiseWeights(L_in,L_out):
        
        epsilon_init = 44
        W =np.random.randint(-1*epsilon_init,epsilon_init,size = ( L_out,1+ L_in))
        W=np.divide(W, 1000)
     
        return W

def sigmoid(X):
        z = 1.0/(1.0 + np.exp(-X))
        return z

def sigmoidGradient(z):
        g=np.multiply( sigmoid(z), (np.ones(z.shape)-sigmoid(z)) )
        return g


def costFunction(nn_params, X, y ,num_labels,lammbda,input_layer_size,hidden_layer_size):
        global counter

        m=y.size

        # initial_Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
        #                 (hidden_layer_size, (input_layer_size + 1)))

        # initial_Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
        #                 (num_labels, (hidden_layer_size + 1)))


        initial_Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
        initial_Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

        J = 0

        eX= np.insert(X, 0, 1, axis=1)
        eXT=np.transpose(eX)

        for i in range(m):
                #a1=np.transpose(np.matrix(X[i]))

                z2 = np.dot(initial_Theta1,eXT[:,i])

                a2=sigmoid(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                hyp=sigmoid(z3)

                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                yt[int(y[i].item())-1] = 1

                # temp = -1*(yt).*log(h) - (ones(num_labels,1) - (yt)).*log(ones(num_labels,1) - h);

                temp = -1*yt
                temp = np.multiply(temp,np.log(hyp))

                temp = temp - np.multiply((np.ones((num_labels,1)) - yt), np.log(np.ones((num_labels,1))- hyp))

                J = J + np.sum(temp)
        

        J = J/m

        # reg = sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));
        #  reg = reg*lambda/m;
        #  reg = reg/2;
         
        #  J = J + reg;

        reg = np.sum( np.multiply(initial_Theta1[:,1:], initial_Theta1[:,1:])) + np.sum( np.multiply(initial_Theta2[:,1:] , initial_Theta2[:,1:]))

        reg = reg*lammbda/m
        reg = reg/2

        J = J + reg

        capdelta1 = np.zeros(initial_Theta1.shape)
        capdelta2 = np.zeros(initial_Theta2.shape)

        eX= np.insert(X, 0, 1, axis=1)
        eXT=np.transpose(eX)

        for i in range(m):
                
                z2 = np.dot(initial_Theta1,eXT[:,i])

                a2=sigmoid(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                hyp=sigmoid(z3)

                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                yt[int(y[i].item())-1] = 1

                delt3=hyp-yt
                delt2=  np.dot(np.transpose(initial_Theta2[:,1:]) ,delt3)

                delt2=np.multiply(delt2,sigmoidGradient(z2))

                capdelta2=capdelta2+ np.dot(delt3,np.transpose(a2))
                capdelta1=capdelta1+ np.dot(delt2,np.transpose(eXT[:,i]))

        Theta1_grad =np.multiply(capdelta1,1/m)
        Theta2_grad =np.multiply(capdelta2,1/m)

        Theta1_grad[:, 1:input_layer_size+1] = Theta1_grad[:, 1:input_layer_size+1] +np.multiply(initial_Theta1[:, 1:input_layer_size+1], (lammbda / m))
        Theta2_grad[:, 1:hidden_layer_size+1] = Theta2_grad[:, 1:hidden_layer_size+1] + np.multiply(initial_Theta2[:, 1:hidden_layer_size+1],(lammbda / m))

        #grad = np.concatenate(( , np.array(Theta2_grad.flatten()) ), axis=1)
        
        
        return J,Theta1_grad,Theta2_grad

def backpropagation(nn_params,X,y,lammbda,num_labels,hidden_layer_size,input_layer_size):
    
        global counter

        m=y.size

        initial_Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
        initial_Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

        capdelta1 = np.zeros(initial_Theta1.shape)
        capdelta2 = np.zeros(initial_Theta2.shape)
        
        Theta1_grad = np.zeros(initial_Theta1.shape)
        Theta2_grad = np.zeros(initial_Theta2.shape)

        eX= np.insert(X, 0, 1, axis=1)
        eXT=np.transpose(eX)

        for i in range(m):
                
                
                
                z2 =np.dot(initial_Theta1,eXT[:,i])
                
                

                a2=sigmoid(z2)

                a2=np.insert(a2, 0, 1, axis=0)

                z3=np.dot(initial_Theta2,a2)

                hyp=sigmoid(z3)

                yt = np.zeros((num_labels,1))

                #print("printing yt nowwwwwwwwwww")
                #print(yt)
                yt[int(y[i].item())-1] = 1

                delt3=hyp-yt
                delt2=  np.dot(np.transpose(initial_Theta2[:,1:]) ,delt3)

                delt2=np.multiply(delt2,sigmoidGradient(z2))

                capdelta2=capdelta2+ np.dot(delt3,np.transpose(a2))
                capdelta1=capdelta1+ np.dot(delt2,np.transpose(eXT[:,i]))

        Theta1_grad =np.multiply(capdelta1,1/m)
        Theta2_grad =np.multiply(capdelta2,1/m)

        Theta1_grad[:, 1:input_layer_size+1] = Theta1_grad[:, 1:input_layer_size+1] +np.multiply(initial_Theta1[:, 1:input_layer_size+1], (lammbda / m))
        Theta2_grad[:, 1:hidden_layer_size+1] = Theta2_grad[:, 1:hidden_layer_size+1] + np.multiply(initial_Theta2[:, 1:hidden_layer_size+1],(lammbda / m))

        #grad = np.concatenate(( , np.array(Theta2_grad.flatten()) ), axis=1)
        
        
        return Theta1_grad,Theta2_grad


initial_Theta1=randinitialiseWeights(num_col,20)
initial_Theta2=randinitialiseWeights(20,10)


initial_nn_params = np.concatenate(( np.array(initial_Theta1.flatten()), np.array(initial_Theta2.flatten()) ))

def computeGradientsCheck(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda):
    paramsInitial = getParams(initial_Theta1,initial_Theta2)
    chkgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4
    hls=15
    ils=100
    ols=10

    backprop_grad1,backprop_grad2=backpropagation(paramsInitial, X, y, lammbda, ols, hls, ils)
    backprop_grad = np.concatenate((np.array( backprop_grad1.flatten()),np.array( backprop_grad2.flatten())),axis=1)

    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        
        initial_Theta1,initial_Theta2= setParams(paramsInitial + perturb,hls,ils,ols)
        
        loss2,_,_ = costFunction(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda,hls)

        initial_Theta1,initial_Theta2= setParams(paramsInitial - perturb,hls,ils,ols)
        lossl,_,_ = costFunction(X, y,initial_Theta1,initial_Theta2,num_labels,lammbda,hls)

        #Compute Check Gradient
        chkgrad[p] = (loss2 - lossl) / (2*e)
        #Return the value we changed to zero:
        perturb[p] = 0

        print("BackpropGrad=%d   CheckGrad=%d",backprop_grad[p],chkgrad[p])
    #Return Params to original value
    

    return chkgrad

counter=0

def gradientDescentnn(X,y,initial_nn_params,alpha,num_iters,lammbda,input_layer_size, hidden_layer_size, num_labels):
    global counter
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    Theta1 = initial_nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = initial_nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    
    m=len(y)
    #J_history =[]
    
    for i in range(num_iters):
        nn_params = np.append(np.array( Theta1.flatten()),np.array( Theta2.flatten()))
        J,grad1, grad2 = fn.costFunction(nn_params, X, y, num_labels, lammbda, input_layer_size, hidden_layer_size)
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        print(J)
        print(" ")
        counter=counter+1
        print(counter)
    
    nn_params = np.concatenate((np.array( Theta1.flatten()),np.array( Theta2.flatten())),axis=1)
    return nn_params

def debugInitializeWeights(fan_out, fan_in):

    w = np.zeros((fan_out, 1 + fan_in))
    # we initialise it with sin to ensure it remains same

    k = 1
    for i in range(fan_out):

        for j in range(fan_in+1):

            w[i, j] = np.sin(k)
            k = k+1
    return w

def computeNumericalGradient(Theta1,Theta2,X_check,y_check,lammbda,num_labels,input_layer_size,hidden_layer_size):
    numgrad1 = np.zeros((Theta1.shape))
    numgrad2 = np.zeros((Theta2.shape))
    perturb1 = np.zeros((Theta1.shape))
    perturb2 = np.zeros((Theta2.shape))
    e = 1e-4

    for i in range(Theta1.shape[0]):
        for j in range(Theta1.shape[1]):

            perturb1[i][j]= e
            #nn_params, X, y ,num_labels,lammbda,input_layer_size,hidden_layer_size

            nn_params1 = np.concatenate(( np.array((Theta1-perturb1).flatten()), np.array((Theta2).flatten()) ))
            nn_params2 = np.concatenate(( np.array((Theta1+perturb1).flatten()), np.array((Theta2).flatten()) ))

            loss1,_ = fn.costFunction(nn_params1,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
            loss2,_ = fn.costFunction(nn_params2,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)

            numgrad1[i][j] = (loss2 - loss1) / (2*e)
            perturb1[i][j] = 0

    for i in range(Theta2.shape[0]):
        for j in range(Theta2.shape[1]):

            perturb2[i][j]= e

            nn_params1 = np.concatenate(( np.array((Theta1).flatten()), np.array((Theta2-perturb2).flatten()) ))
            nn_params2 = np.concatenate(( np.array((Theta1).flatten()), np.array((Theta2+perturb2).flatten()) ))


            loss1,_ = fn.costFunction(nn_params1,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
            loss2,_ = fn.costFunction(nn_params2,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
            numgrad2[i][j] = (loss2 - loss1) / (2*e)
            perturb2[i][j] = 0

    return numgrad1, numgrad2
    


def check_gradients(lammbda):

    input_layer_size = 10
    hidden_layer_size = 3
    num_labels = 10
    m = 10

    ##generating random test data 

    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    ##generating X using same function

    X_check = debugInitializeWeights(m, input_layer_size-1)
    #y  = 1 + np.transpose( np.mod(1:m, num_labels) )
    y_check = np.zeros((m,1))
    for i in range(m):
        y_check[i] = ((i+1)%num_labels) +1 

    nn_params = np.concatenate(( np.array(Theta1.flatten()), np.array(Theta2.flatten())), axis=0)

    #cost = fn.costFunction(nn_params,X_check,y_check,num_labels,lammbda,input_layer_size,hidden_layer_size)
    
    Xc_ones=np.insert(X_check, 0, 1, axis=1)

    #grad1,grad2 = costFunction(nn_params, X_check, y_check, num_labels, lammbda, input_layer_size, hidden_layer_size)
    #numgrad = computeNumericalGradient(costFunc, nn_params);
    grad1,grad2 = costFunction(nn_params, X_check, y_check, num_labels, lammbda, input_layer_size, hidden_layer_size)
    
    numgrad1,numgrad2 = gradientDescentnn(X_checl, y_check, nn_params, 0.8, 200, 0, input_layer_size, hidden_layer_size, num_labels)

    grad = np.concatenate( ( np.array(grad1.flatten()), np.array(grad2.flatten()) ), axis=1)
    
    numgrad = nn_params = np.concatenate(( np.array(numgrad1.flatten()), np.array(numgrad2.flatten()) ), axis=0)
    # numgrad=[numgrad]
    # numgrad=np.array(numgrad)

    #diff = np.linalg.norm(numgrad-grad)/ np.linalg.norm(numgrad+grad)

    # print("if gradients are correct, diff should be less than 1e-9")
    # print("diff: ",diff)

    #return diff

    for i in range(len(numgrad)):
        print("Numerical Gradient = %f. BackProp Gradient = %f."%(grad[i],numgrad[i]))


#nn_params=computeGradientsCheck(X, y, initial_Theta1, initial_Theta2, num_labels, lammbda)

nn_params=gradientDescentnn(X, y, initial_nn_params, alpha, num_iters, lammbda, input_layer_size, hidden_layer_size, num_labels)