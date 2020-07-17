#import os
#import ipdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error

class NeuralNetwork:
    '''simple feedforward neural network class'''       
    def __init__(self,num_inputs,num_outputs,num_neurons,tf_functions,epochs=100,learning_rate=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_neurons = num_neurons

        self.min_weight_value = -1
        self.max_weight_value = 1
        self.tf_functions = []
        #self.tf_functions_derivatives = []

        for i in range(0,len(tf_functions)):
            if tf_functions[i] == "logistic":
                self.tf_functions.append(self.tf_logistic)
                #self.tf_functions_derivatives.append(self.tf_logistic_derivative)                
            elif tf_functions[i] == "linear":
                self.tf_functions.append(self.tf_linear)
                #self.tf_functions_derivatives.append(self.tf_linear_derivative)
            else:
                print("Unknown transfer function: %s" %(tf_function[i]))
                exit()
        self.outputs = None # will be updated after calculation is called

        #********************************
        # Initialize weights and bias
        #********************************
        weights_per_layer = []
        biases_per_layer = []
        wsums_per_layer = []  # for helping later activation function approximate derivative calculation
        for l in range(0,len(num_neurons)): # for all layers
            if (l == 0):
                previous_layer_size = len(inputs[0])
            else:
                previous_layer_size = num_neurons[l-1]
            layer_size = num_neurons[l]
            layer_weights = np.zeros((layer_size,previous_layer_size))
            weights_per_layer.append(layer_weights)
            biases_per_layer.append(np.zeros(layer_size))
            wsums_per_layer.append(np.zeros(layer_size))

        self.weights_per_layer = weights_per_layer
        self.biases_per_layer = biases_per_layer    
        self.wsums_per_layer = wsums_per_layer

        #self.randomize_weights()
        #self.randomize_biases()


    def set_last_layer_weights(self,weights):
        self.weights_per_layer[-1] = [weights]

    def randomize_weights(self):
        a = self.min_weight_value
        b = self.max_weight_value
        for i, layer in enumerate(self.weights_per_layer):
            #self.weights_per_layer[i] = np.random.random(self.weights_per_layer[i].shape)
            self.weights_per_layer[i] = a+(b-a)*np.random.random(self.weights_per_layer[i].shape)

    def randomize_biases(self):
        a = self.min_weight_value
        b = self.max_weight_value
        for i, layer in enumerate(self.biases_per_layer):
            self.biases_per_layer[i] = a+(b-a)*np.random.random(self.biases_per_layer[i].shape)

    def predict(self,X):
        Ypredicted = []
        for i in range(0,X.shape[0]):  #for every pattern 
            predicted = model.calc_output(X[i]) # calculating outputs for a given pattern
            Ypredicted.append(predicted)
        return np.array(Ypredicted)

    def calc_output(self,inputs):
        '''calculates the output of the neural network'''
        #*****************
        # creates empty output array
        #*****************
        outputs = []
        for i in range(0,len(self.num_neurons)):
            outputs.append(np.zeros(num_neurons[i])) 

        outputs = np.array(outputs) #convert to numpy array
        #*****************
        # calculates output for each layer
        #*****************
        for i in range(0,len(self.num_neurons)):
            if (i == 0): # first layer
                outputs[i] = self.calc_layer(inputs,self.weights_per_layer[i],self.biases_per_layer[i],self.tf_functions[i],i)
            else:
                outputs[i] = self.calc_layer(outputs[i-1],self.weights_per_layer[i],self.biases_per_layer[i],self.tf_functions[i],i)
        self.outputs = outputs
        return outputs[-1]

    def calc_layer(self,inputs,weights,biases,tf_function,layer_index):
        outputs = np.zeros(len(weights))
        for i in range(0,len(weights)): #for all neurons
            outputs[i] = self.neuron(weights[i],biases[i],inputs,tf_function,layer_index,i)
        return outputs

    def tf_logistic(self,x):
        return 1/(1+np.exp(-x*2))
        #return 1/(1+np.exp(-x))

    def tf_logistic_derivative(self,y):        
        return (1-y)*(y) 

    def tf_linear(self,x):
        return x
    def tf_linear_derivative(self,y):
        return 1

    def neuron(self,weights,bias,inputs,tf_function,layer_index,neuron_index):
        weighted_sum = np.dot(weights,inputs)+bias
        output = tf_function(weighted_sum)

        # update for later calculate activation function approximate derivative
        self.wsums_per_layer[layer_index][neuron_index] = weighted_sum

        return output
    def fit(self,X,Y):

        # sequential approach
        self.ssr_total_list = []
        self.mse_total_list = []
        for epc in range(0,self.epochs+1):
            ssr_total = 0
            mse_total = 0
            idxlist = np.arange(0,X.shape[0])
            #on first evaluation, local gradients are not updated, just to save the initial solution            
            np.random.shuffle(idxlist)  #randomize index_list            
            for i in range(0,X.shape[0]):  
                predicted = model.calc_output(X[idxlist[i]]) # calculating outputs for a given pattern  
                output_error = Y[idxlist[i]] - predicted  # error for each output
                ssr = 0.5*sum(output_error**2) #sum of squared residuals
                ssr_total += ssr
                mse_total += sum(output_error**2)
                # calculate local gradients
                if (epc != 0):
                    self.calculate_local_gradients(output_error,ssr)
                    self.update_weights(X[idxlist[i]])
            mse = mse_total/X.shape[0]
            self.ssr_total_list.append(ssr_total)
            self.mse_total_list.append(mse)
            #print("Epoch: %d \t SSR_total = %f" %(epc,ssr_total))



    def calculate_local_gradients(self,output_error,ssr):
        ''' calculates local gradients '''
        # initializes local gradients
        self.local_gradients = [0]*len(num_neurons) # will be a list for each element
        for i in range(0,len(self.local_gradients)):
            self.local_gradients[i] = np.zeros(num_neurons[i])

        # calculates local gradients for output layer
        for j in range(0,self.num_neurons[-1]): # for all neurons in the output layer
            partiald_E_y = -(output_error[j])
            #local_gradient = -partiald_E_y * self.tf_functions_derivatives[-1](self.outputs[-1][j])
            tf = self.tf_functions[-1]
            local_gradient = -partiald_E_y * self.num_derivative(tf,self.wsums_per_layer[-1][j])
            self.local_gradients[-1][j] = local_gradient

        # calculates local gradients for hidden layers
        for l in range(len(num_neurons)-2,-1,-1):  # for all hidden layers, from last to first
            for j in range(0,num_neurons[l]):
                outsum = 0
                w_from_this_neuron_to_next_layer = self.weights_per_layer[l+1][:,j]
                for o in range(0,num_neurons[l+1]): # for all neurons on the next layer
                    wok = w_from_this_neuron_to_next_layer[o]
                    outsum+= self.local_gradients[l+1][o]*wok                
                #self.local_gradients[l][j] = self.tf_functions_derivatives[l](self.outputs[l][j])*outsum
                tf = self.tf_functions[l]
                self.local_gradients[l][j] = self.num_derivative(tf,self.wsums_per_layer[l][j])*outsum

        return 1
    def update_weights(self,inputs):
        '''update weights and bias for backpropagation'''
        # weight update
        for l in range(0,len(num_neurons)): # for all layers
            for j in range(0,num_neurons[l]): # for all neurons in layer
                if (l == 0): # first hidden layer
                    for p in range(0,self.num_inputs): # for all inputs
                        self.weights_per_layer[l][j,p] += self.learning_rate * self.local_gradients[l][j] * inputs[p]
                else:
                    for p in range(0,num_neurons[l-1]): # for all neurons in previous layer
                        self.weights_per_layer[l][j,p] += self.learning_rate * self.local_gradients[l][j] * self.outputs[l-1][p]

                # bias update
                self.biases_per_layer[l][j] += self.learning_rate * self.local_gradients[l][j] * 1
        
        print("Biases per layer: ", self.biases_per_layer)
        print("Pesos por layer: ", self.weights_per_layer)
        print("Local Gradients: ", self.local_gradients)
        
    def num_derivative(self,f,x,delta=1e-6):
        return (f(x+delta)-f(x))/delta


inputs = np.array([[0,1]])
outputs = np.array([1])

num_outputs = np.size(outputs[0]) ## get the number of outputs of the neural network
num_inputs = np.size(inputs[0])

num_neurons = np.array([2, num_outputs])
num_layers = 1

np.random.seed(0)
model = NeuralNetwork(num_inputs, num_outputs, num_neurons, ["logistic", "linear"], epochs = 1, learning_rate = 0.1)

model.fit(inputs,outputs)

model.calc_output(inputs[0])