# Basic Perceptron

'''This is a Basic neural network with no hidden layers, a perceptron,
to solve the following problem. Given the following, NN should predict 
what new is.

        inputs          Outputs
Ex.1    |0  |0  |1  |   -> 0
Ex.2    |1  |1  |1  |   -> 1
Ex.3    |1  |0  |1  |   -> 1
Ex.3    |0  |1  |1  |   -> 0

New    |1  |0  |0  |   -> ?

Normalise(Sum of X_i*W_i) a normalised sum of weighted inputs
inputs = neural stimulus
weights = synapse response of neuron
normalised sum = the neuronal response to stimulus

we will be using the sigmoid function
------------------------------------------------'''
#Code

import numpy as np

# define the sigmoid function
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#define the sigmoid derivative
def sigmoid_derivative(x):
    return x * (1-x)

# define the training inputs

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# define the corresponding output vector (which we have to transpose to form a 4*1 mat)
training_outputs = np.array([[0,1,1,0]]).T

#seed random numbers

np.random.seed(1)

#a 3*1 matrix with values -1 to 1 with mean 0
synaptic_weights = 2 * np.random.random((3,1))-1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# main loop

for iteration in range(100000):

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer,synaptic_weights))

    #backpropagation

    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic Weights after training: ')
print(synaptic_weights)

print('outputs after training: ')
print(outputs)

#Training process
#1 take inputs from training examle and pu them through the formula
#to get the neurons output
#2 calculate the error, the difference of neuron output and actual output
#3 depending of the severness of the error adjust weights accoringly
#4 repeat 20,000 times

#Backpropagation
#Error weighted derivative
# error * input * (sigmoid(output))'


