import numpy as np
import matplotlib.pyplot as plt

def hardlim(z):
	for i in range(3):
		if z[i]>=0:
			z[i] = 1
		else:
			z[i] = 0
	return z

class Perceptron():
	def __init__(self):
		self.biases = np.random.randn(3, 1)
		self.weights = np.random.randn(3, 2)
		self.trainging_data = []

	def feedforward(self, x):
		return hardlim(np.dot(self.weights,x) + self.biases)

	def learning(self, alpha):
		trainging_data = self.trainging_data
		itr = 0
		while True:
			itr = itr + 1
			error = 0.0
			for p, t in trainging_data:
				a = hardlim(np.dot(self.weights,p) + self.biases)
				e = t - a
				self.weights = self.weights + \
					np.dot((e*alpha),p.transpose())
				self.biases =self.biases + (e*alpha)
				error = error + (e**2).sum()/2
			if error < 0.01 or itr>1000:
				return itr

	def plot(self):
		x_axis = np.arange(-3.,3.,0.1)
		y_axis = np.arange(-3.,3.,0.1)
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
		x3 = [in1 for (in1,in2),out in data if out==2]
		y3 = [in2 for (in1,in2),out in data if out==2]

		ax = plt.gca()
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.xaxis.set_ticks_position('bottom')
		ax.spines['bottom'].set_position(('data',0))
		ax.spines['bottom'].set_linewidth(2)
		ax.spines['left'].set_linewidth(2)
		ax.yaxis.set_ticks_position('left')
		ax.spines['left'].set_position(('data',0))

		plt.plot(x0,y0,'co',x1,y1,'mo',x3,y3,'go')
		plt.axis('equal')
		plt.show()
	
	def load_data(self,trainging_data):
		self.trainging_data = []
		for inpt, t in trainging_data:
			x = np.array(inpt).reshape(2,1)
			y = np.array(t).reshape(3,1)
			self.trainging_data.append((x,y))

def scalarize(y):
	return np.argmax(y)
def arraylize(x):
	return [a for a in x.flat]
				
def plot(xs,ys):
	plt.plot(xs,ys)
	plt.show()
	
def main():
	trainging_data = [([1,1],[1,0,0]),([0,2],[1,0,0]),([3,1],[1,0,0]),\
		([2,-1],[0,1,0]),([2,0],[0,1,0]),([1,-2],[0,1,0]),\
		([-1,2],[0,0,1]),([-2,1],[0,0,1]),([-1,1],[0,0,1]),]
	# xs = []
	# ys = []
	# for i in np.arange(0.01,1.01,0.01):
	# 	result = []
	# 	for j in range(20):
	# 		pcptrn = Perceptron()
	# 		pcptrn.load_data(trainging_data)
	# 		result.append(pcptrn.learning(i))
	# 	print("{}:\t{}".format(i, max(result)))
	# 	xs.append(i)
	# 	ys.append(max(result))
	# plot(xs,ys)
	
	pcptrn = Perceptron()
	pcptrn.load_data(trainging_data)
	itr = pcptrn.learning(0.8)
	pcptrn.plot()
if __name__ == '__main__':
	main()


