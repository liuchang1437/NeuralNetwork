import numpy as np
import matplotlib.pyplot as plt
import random

sigma_init = 1.5
mu_init = 2
learning_rate = 0.2

class SOM():
	def __init__(self,x,y):
		# 1. initializing the weights
		self.weights = [[sigma_init * np.random.randn(1,2) + sigma_init for i in range(x)] for j in range(y)] 
		self.r0 = float(max(x,y))/2
	
	def train(self,data,num_itr):
		size = len(data)
		lameda = float(num_itr)/np.log(sigma_init)
		for itr in range(num_itr):
			BMU = 9999.9
			BMU_x = 0
			BMU_y = 0
			# select an input vector randomly
			input_vector = data[np.random.randint(0,size)]
			# 2. calculating the best matching unit
			for i in range(5):
				for j in range(5):
					dist = np.sqrt(np.sum(np.square(self.weights[i][j]-input_vector)))
					if dist < BMU:
						BMU = dist
						BMU_x = i
						BMU_y = j
			# 3. determine the BMU's local neighbourhood
			radius = self.r0*np.exp(-float(itr)/lameda)

			# 4. adjusting the weights
			for i in range(5):
				for j in range(5):
					# circle
					# d_BMU = np.sqrt((i-BMU_x)**2 + (j-BMU_y)**2)
					# rectangular
					d_BMU = max(abs(i-BMU_x),abs(j-BMU_y))
					if d_BMU <= radius:
						Theta = np.exp(-d_BMU/2/(radius**2))
						current_learning_rate = learning_rate * np.exp(-float(itr)/lameda)
						self.weights[i][j] = self.weights[i][j] + Theta * current_learning_rate * (input_vector-self.weights[i][j])

	
def plot(xs,ys,rowX,rowY,colX,colY):
	plt.plot(xs[:300],ys[:300],'ro')
	plt.plot(xs[300:],ys[300:],'g+')
	plt.plot(rowX,rowY,'b-')
	plt.plot(colX,colY,'b-')
	#plt.show()
	plt.savefig('rect_result_ita_{}_sigma_{}.png'.format(learning_rate,sigma_init))

def main():
	train = open("data/data_processed.txt")
	train_data = []
	xs = []
	ys = []

	for line in train:
		x, y = line.split()
		x = float(x)
		y = float(y)
		xs.append(x)
		ys.append(y)
		train_data.append(np.array([x,y]).reshape(1,2))

	s = SOM(5,5)
	s.train(train_data,100000)

	rowX = [[] for i in range(5)]
	rowY = [[] for j in range(5)]
	colX = [[] for i in range(5)]
	colY = [[] for j in range(5)]
	for i in range(5):
		for j in range(5):
			rowX[i].append(s.weights[i][j][0][0])
			rowY[i].append(s.weights[i][j][0][1])
			colX[j].append(s.weights[i][j][0][0])
			colY[j].append(s.weights[i][j][0][1])
	plot(xs,ys,rowX,rowY,colX,colY)

if __name__ == '__main__':
	main()