import random
import numpy as np
import matplotlib.pyplot as plt

class MLQP():
	""" It's similar with a tradition backprop algoritm, except that
	the activation function is y_j = sum_i(u_j * (y_i^2) + v_j * y_i) + b_j
	"""
	def __init__(self, sizes):
		""" The network consists of len(sizes) layers, 
		each of which contains sizes[i] neurons."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.u = [np.random.randn(j,i) for i, j in zip(sizes[:-1],sizes[1:])]
		self.v = [np.random.randn(j,i) for i, j in zip(sizes[:-1],sizes[1:])]
		self.b = [np.random.randn(x,1) for x in sizes[1:]]

	def feedforward(self, x):
		""" Return the output when input is equal to x"""
		for u, v, b in zip(self.u, self.v, self.b):
			x = sigmoid(np.dot(u,x**2) + np.dot(v,x) + b)
		return x

	def SGD_online(self, training_data, eta):
		""" on-line mode, update weights every single input."""
		print("Execute in on-line mode..")
		itr = 0
		while True:
			itr = itr+1
			random.shuffle(training_data)
			error_total = self.update_online(training_data, eta)

			# the MSE of output 
			if error_total < 0.01:
				print("Iteration times: {}".format(itr))
				return itr
			if itr%1000==0:
				print(itr)
			if itr>10000:
				return itr

	def update_online(self, training_data, eta):
		error_total = 0
		for x, y in training_data:
			nabla_u, nabla_v, nabla_b, error_single = self.backprop(x, y)

			self.u = [u-(eta*nu) for u, nu in zip(self.u, nabla_u)]
			self.v = [v-(eta*nv) for v, nv in zip(self.v, nabla_v)]
			self.b = [b-(eta*nb) for b, nb in zip(self.b, nabla_b)]
			error_total = error_total+error_single

		return error_total/len(training_data)

	def SGD_batch(self, training_data, mini_batch_size, eta):
		""" batch mode, update weights every epoch."""
		print("Execute in batch mode..")
		num_data = len(training_data)
		itr = 0
		while True:
			itr = itr + 1
			random.shuffle(training_data)
			mini_batches = [ training_data[k:k+mini_batch_size] \
							for k in xrange(0,num_data,mini_batch_size)]
			error_total = 0.0
			for mini_batch in mini_batches:
				error_total = error_total + self.update_mini_batch(mini_batch, eta)
			
			# the MSE of output 
			if error_total/num_data < 0.01:
				print("Iteration times: {}".format(itr))
				return itr
			if itr%1000==0:
				print(itr)
			if itr>10000:
				return itr

	def update_mini_batch(self, mini_batch, eta):
		nabla_u = [np.zeros(u.shape) for u in self.u]
		nabla_v = [np.zeros(v.shape) for v in self.v]
		nabla_b = [np.zeros(b.shape) for b in self.b]

		error_total = 0.0

		for x, y in mini_batch:
			delta_nabla_u, delta_nabla_v, delta_nabla_b, error_single = self.backprop(x, y)
			nabla_u = [nu+dnu for nu, dnu in zip(nabla_u, delta_nabla_u)]
			nabla_v = [nv+dnv for nv, dnv in zip(nabla_v, delta_nabla_v)]
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			error_total = error_total + error_single
				
		self.u = [u-(eta/len(mini_batch)*nu) for u, nu in zip(self.u, nabla_u)]
		self.v = [v-(eta/len(mini_batch)*nv) for v, nv in zip(self.v, nabla_v)]
		self.b = [b-(eta/len(mini_batch)*nb) for b, nb in zip(self.b, nabla_b)]

		return error_total


	def backprop(self, x, y):
		nabla_u = [np.zeros(u.shape) for u in self.u]
		nabla_v = [np.zeros(v.shape) for v in self.v]
		nabla_b = [np.zeros(b.shape) for b in self.b]

		# feedforward
		activation = x
		zs = []
		activations = [x]
		for u, v, b in zip(self.u, self.v, self.b):
			z = np.dot(u,activation**2) + np.dot(v, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_u[-1] = np.dot(delta, (activations[-2]**2).transpose())
		nabla_v[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			
			delta = np.dot(((2*self.u[-l+1]).transpose()*activations[-l]+self.v[-l+1].transpose()), delta) * sp
			nabla_b[-l] = delta
			nabla_u[-l] = np.dot(delta, (activations[-l-1]**2).transpose())
			nabla_v[-l] = np.dot(delta, activations[-l-1].transpose())
		
		return (nabla_u, nabla_v, nabla_b, ((activations[-1]-y)**2).sum())
	
	def evaluate(self, test_data):
		test_results = [(scalarize(self.feedforward(x)),y) \
						for x,y in test_data]
		num_success = sum(int(x==y) for (x,y) in test_results)
		print("correct/total: {}/{} ({:.2f}%)".format(num_success,\
			len(test_data),float(num_success)/len(test_data)*100))
		return float(num_success)/len(test_data)*100
	
	def plot(self):
		"""Draw the class distribution"""
		x_axis = np.arange(-3.,3.,0.01)
		y_axis = np.arange(-3.,3.,0.01)
		data = []
		for x in x_axis:
			for y in y_axis:
				data.append(np.array([x,y]).reshape(2,1))
		data = [ (arraylize(inpt),scalarize(self.feedforward(inpt))) \
				for inpt in data]
		x0 = [in1 for (in1,in2),out in data if out==0]
		y0 = [in2 for (in1,in2),out in data if out==0]
		x1 = [in1 for (in1,in2),out in data if out==1]
		y1 = [in2 for (in1,in2),out in data if out==1]
		plt.plot(x0,y0,'co',x1,y1,'mo')
		plt.axis('equal')
		plt.show()

# utils
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
def scalarize(y):
	return np.argmax(y)
def arraylize(x):
	return [a for a in x.flat]