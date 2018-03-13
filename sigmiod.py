import numpy as np

def sigmoid(x):
	return 1.0/(1.0+ np.exp(x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1.0-x**2

class NeuralNetworks:
	def __init__ (self,layers,activation='tanh'):
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime 

		elif activation == 'tanh':
			self.activation= tanh
			self.activation_prime = tanh_prime

		# Set weights
		self.weights = []
		layers=[2,2,1]
		#range of weights value(-1,1)
		#input and hidden layers-random((2+1,2+1)):3x3

		for i in range(1, len(layers)-1):
			r = 2*np.random.random((layers[i-1]+1,layers[i]+1)) -1
			self.weights.append(r)

		#output layers -rand((2+1,1))3x1

		r = 2*np.random.random((layers[i]+1,layers[i+1])) -1
		self.weights.append(r)


	def fit(self,X,y,learning_rate=0.2,epochs=1000000):
		#Add column of ones to X
		#This is to add the bias unit to the input layer
		ones = np.atleast_2d(np.ones(X.shape[0]))
		X = np.concatenate((ones.T,X),axis=1)

		for k in range(epochs):
			i =np.random.randint(X.shape[0])
			a = [X[i]]

			for l in range(len(self.weights)):
				dot_value = np.dot(a[l],self.weights[l])
				activation= self.activation(dot_value)
				a.append(activation)

			#output layer
			error = y[i] - a[-1]
			deltas = [error*self.activation_prime(a[-1])]

			#we need to begin at th second to last
			#(a layer before the output layer)

			for l in range(len(a)-2,0,-1):
				deltas.append(deltas[-1].dot(self.weights[i].T)*self.activation_prime(a[l]))

			#reverse
			#

