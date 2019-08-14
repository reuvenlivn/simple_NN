import numpy as np


#1. Construct a function returning a sigmoid function
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

#2. Construct a function returning a sigmoid derivative function
def sigmoid_derivative(x):
    return (x * (1 - x))

# one layer implementation

# initialize weights
np.random.seed(1)
w = 2 * np.random.random((3,1)) - 1

#3. Build an array of three weights and initialize them randomly
# input data
x = np.array([[0,0,5], 
              [0,5,-5], 
              [5,0,5], 
              [5,-5,5]])
    
# output data
y = np.array([0,1,1,0])

#4. Build a loop iterating 1000 times (the number of learning ste יps we are using)
#5. For every iteration calculate the error and then update the corresponding weight
for i in range(1000):
    l1 = sigmoid(np.dot(x, w))
#    print('l1={}'.format(l1))

    error = y - l1.T
#    print('error={}'.format(error))

    l1_delta = error.T * sigmoid_derivative(l1)
#    print('l1_delta={}'.format(l1_delta))

    w = w + np.dot(x.T, l1_delta)
#    print('weights={}'.format(weights))

print('l1={}'.format(l1))

#For a slightly harder problem we will this time add another layer to the problem, a hidden
#layer. It will have three neurons in it. The output layer will have one neuron as
#previously. Build the for loop for updating the weights, this time it updates weights of two
#layers.

# initialize weights
np.random.seed(1)
w1 = 2 * np.random.random((3,4)) - 1
#w2 = 2 * np.random.random((4,4)) - 1
w2 = 2 * np.random.random((4,1)) - 1

# input data
x = np.array([[0,0,1], 
              [0,1,1], 
              [1,0,1], 
              [1,1,1]])
#output data
y = np.array([0,1,1,0])

# initialize weights
np.random.seed(1)
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1

# Create data
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([0,0,1,1])

# forward propegate
for i in range(1000):
    l1 = sigmoid(np.dot(x,w0))
    l2 = sigmoid(np.dot(l1,w1))

    l2_error = y - l2.T
    l2_delta = l2_error.T * sigmoid_derivative(l2)
    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * sigmoid_derivative(l1)

    w1 += np.dot(l1.T, l2_delta)
    w0 += np.dot(x.T, l1_delta)

print('l2={}'.format(l2))

## Two layers implementation

# initialize weights
#np.random.seed(1)
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,3)) - 1
w2 = 2 * np.random.random((3,4)) - 1
w3 = 2 * np.random.random((4,1)) - 1

# Create data
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([0,0,1,1])

# forward propegate
for i in range(10000):
    l1 = sigmoid(np.dot(x,w0))
    l2 = sigmoid(np.dot(l1,w1))
    l3 = sigmoid(np.dot(l2,w2))
    l4 = sigmoid(np.dot(l3,w3))

    l4_error = y - l4.T
    l4_delta = l4_error.T * sigmoid_derivative(l4)

    l3_error = np.dot(l4_delta, w3.T)
    l3_delta = l3_error.T * sigmoid_derivative(l3)

    l2_error = np.dot(l3_delta, w2.T)
    l2_delta = l2_error * sigmoid_derivative(l2)
       
    l1_error = np.dot(l2_delta, w1.T)
    l1_delta = l1_error * sigmoid_derivative(l1)

    w3 += np.dot(l3.T, l4_delta)
    w2 += np.dot(l2.T, l3_delta)
    w1 += np.dot(l1.T, l2_delta)
    w0 += np.dot(x.T, l1_delta)

print('l4={}'.format(l4))

#Create a neural network with one hidden layer for training the famous iris dataset. Play
#with the number of hidden layers and try to improve the percentage of success.

#
#from sklearn.datasets import load_iris
##
#iris = load_iris()
#X = 0.1 * iris.data
#Y = (0.5 * iris.target)**2
#Y = Y.reshape((150,1))
##Y = np.reshape(Y,(150,1))
#data_size = Y.size
#
## initialize weights
#np.random.seed(1)
##np.random.seed(1)
#W0 = 2 * np.random.random((4,data_size)) - 1
#W1 = 2 * np.random.random((data_size,4)) - 1
#W2 = 2 * np.random.random((4,data_size)) - 1
#W3 = 2 * np.random.random((data_size,1)) - 1
#
## forward propegate
#for i in range(2):
#    L1 = sigmoid(np.dot(X,W0))
#    L2 = sigmoid(np.dot(L1,W1))
#    L3 = sigmoid(np.dot(L2,W2))
#    L4 = sigmoid(np.dot(L3,W3))
#
#    L4_error = Y - L4
#    L4_delta = L4_error * sigmoid_derivative(L4)
#
#    L3_error = np.dot(L4_delta, W3.T)
#    L3_delta = L3_error * sigmoid_derivative(L3)
#
#    L2_error = np.dot(L3_delta, W2.T)
#    L2_delta = L2_error * sigmoid_derivative(L2)
#       
#    L1_error = np.dot(L2_delta, W1.T)
#    L1_delta = L1_error * sigmoid_derivative(L1)
#
#    W3 += np.dot(L3.T, L4_delta)
#    W2 += np.dot(L2.T, L3_delta)
#    W1 += np.dot(L1.T, L2_delta)
#    W0 += np.dot(X.T, L1_delta)
#
#print('L4={}'.format(L4))