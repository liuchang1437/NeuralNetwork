import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import plotly
import plotly.graph_objs as go

sigma_init = 7
mu_init = 2
learning_rate = 0.2

class SOM():
	def __init__(self,x,y):
		# 1. initializing the weights
		self.weights = [[sigma_init * np.random.randn(1,310) + mu_init for i in range(y)] for j in range(x)]
		self.r0 = float(max(x,y))/2
		self.x = x
		self.y = y
	
	def train(self,data,num_itr):
		cnt = 0
		size = len(data)
		lameda = float(num_itr)/np.log(sigma_init)
		
		for itr in range(num_itr):
			print(cnt)
			cnt = cnt + 1
			BMU = 9999999.9
			BMU_x = 0
			BMU_y = 0
			# select an input vector randomly
			input_vector = data[np.random.randint(0,size)]
			# 2. calculating the best matching unit
			for i in range(14):
				for j in range(21):
					dist = np.sqrt(np.sum(np.square(self.weights[i][j]-input_vector)))
					if dist < BMU:
						BMU = dist
						BMU_x = i
						BMU_y = j

			# 3. determine the BMU's local neighbourhood
			radius = self.r0*np.exp(-float(itr)/lameda)

			# 4. adjusting the weights
			for i in range(14):
				for j in range(21):
					# circle
					d_BMU = np.sqrt((i-BMU_x)**2 + (j-BMU_y)**2)
					# rectangular
					# d_BMU = max(abs(i-BMU_x),abs(j-BMU_y))
					if d_BMU <= radius:
						Theta = np.exp(-d_BMU/2/(radius**2))
						current_learning_rate = learning_rate * np.exp(-float(itr)/lameda)
						self.weights[i][j] = self.weights[i][j] + Theta * current_learning_rate * (input_vector-self.weights[i][j])
			# end adjusting the  weights
		# end for itr

		# 5. plot
		BMU_cnt = [[0 for i in range(self.y)] for j in range(self.x)]
		for i in range(size):
			input_vector = data[i]
			BMU = 9999999.9
			BMU_x = 0
			BMU_y = 0
			for i in range(14):
				for j in range(21):
					dist = np.sqrt(np.sum(np.square(self.weights[i][j]-input_vector)))
					if dist < BMU:
						BMU = dist
						BMU_x = i
						BMU_y = j
			BMU_cnt[BMU_x][BMU_y] = BMU_cnt[BMU_x][BMU_y] + 1
		print(BMU_cnt)

		self.plot(BMU_cnt)

	
	def plot(self,BMU_cnt):
		trace = go.Heatmap(z=BMU_cnt,colorscale="Blackbody")
		data=[trace]
		plotly.offline.plot(data, filename='basic-heatmap')

def main():
	train = sio.loadmat('data/hw4-EEG.mat')['EEG_X']
	train_data = []
	for data in train:
		train_data.append(np.array(data).reshape(1,310))

	s = SOM(14,21)
	s.train(train_data,100000)



if __name__ == '__main__':
	main()