import loader
import numpy as np
import matplotlib.pyplot as plt
import random
import time

#from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("..")
from MLQP import MLQP

def main():
	###############  random loader, plot classes distribution ############
	# training_data, test_data = loader.random_load_data()
	# net = min_max([2,10,2],0.8)
	# print(net.learning(training_data))
	# net.evaluate(test_data)
	# net.plot("min_max")

	# ###############  random loader, evaluate performance ############
	# training_data, test_data = loader.random_load_data()
	# test_performance(training_data, test_data, 0.1, 1.2, 0.05)

	# =========== test time ======================
	training_data, test_data = loader.random_load_data()
	result = []
	itrs = []
	for i in range(100):
		net = min_max([2,10,2],0.8)
		itr,tt = net.learning(training_data) # (., epochs, batch size, learning rate)
		result.append(tt)
		itrs.append(itr)
		#net.evaluate(test_data)
	print("Total time: {:.2f}".format(sum(result)/100))
	print("Iteration: {:.2f}".format(sum(itrs)/100))


class min_max():
	def __init__(self, sizes, eta):
		""" The network consists of len(sizes) layers, 
		each of which contains sizes[i] neurons."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.net11 = MLQP.MLQP(sizes)
		self.net12 = MLQP.MLQP(sizes)
		self.net01 = MLQP.MLQP(sizes)
		self.net02 = MLQP.MLQP(sizes)
		self.eta = eta

	def learning(self,training_data):
		train_time = []
		itrs = []

		begin = time.clock()
		itrs.append(self.net11.SGD_online(training_data[0], self.eta))
		end = time.clock()
		train_time.append(end - begin)

		begin = time.clock()
		itrs.append(self.net12.SGD_online(training_data[1], self.eta))
		end = time.clock()
		train_time.append(end - begin)

		begin = time.clock()
		itrs.append(self.net01.SGD_online(training_data[2], self.eta))
		end = time.clock()
		train_time.append(end - begin)

		begin = time.clock()
		itrs.append(self.net02.SGD_online(training_data[3], self.eta))
		end = time.clock()
		train_time.append(end - begin)
		avg_time = (sum(train_time)-max(train_time)-min(train_time)) / 2
		avg_itr = (sum(itrs)-max(itrs)-min(itrs)) / 2
		print("Training time: {:.2f} s".format(avg_time))

		return avg_itr,avg_time
	
	def MIN(self,x1,x2):
		if x1==1 and x2==1:
			return 1
		return 0
	
	def MAX(self,x1,x2):
		if x1==0 and x2==0:
			return 0
		return 1
	
	def feedforward(self, x):
		min1 = self.MIN(self.net11.evaluate_single(x), self.net12.evaluate_single(x))
		min2 = self.MIN(self.net01.evaluate_single(x), self.net02.evaluate_single(x))
		result = self.MAX(min1, min2)
		return result

	def evaluate(self, test_data):
		test_results = [(self.feedforward(x),y) \
						for x,y in test_data]
		num_success = sum(int(x==y) for (x,y) in test_results)
		print("correct/total: {}/{} ({:.2f}%)".format(num_success,\
			len(test_data),float(num_success)/len(test_data)*100))
		return float(num_success)/len(test_data)*100

	def plot(self,filename):
		"""Draw the class distribution"""
		x_axis = np.arange(-3.,3.,0.01)
		y_axis = np.arange(-3.,3.,0.01)
		data = []
		for x in x_axis:
			for y in y_axis:
				data.append(np.array([x,y]).reshape(2,1))
		data = [ (MLQP.arraylize(inpt),self.feedforward(inpt)) \
				for inpt in data]
		x0 = [in1 for (in1,in2),out in data if out==0]
		y0 = [in2 for (in1,in2),out in data if out==0]
		x1 = [in1 for (in1,in2),out in data if out==1]
		y1 = [in2 for (in1,in2),out in data if out==1]
		ax = plt.gca()
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.xaxis.set_ticks_position('bottom')
		ax.spines['bottom'].set_position(('data',0))
		ax.yaxis.set_ticks_position('left')
		ax.spines['left'].set_position(('data',0))
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.plot(x0,y0,'co',label='class1')
		plt.plot(x1,y1,'mo',label='class2')
		plt.axis('equal')
		plt.savefig('{}.png'.format(filename))
		self.net01.plot("net01")
		self.net02.plot("net02")
		self.net11.plot("net11")
		self.net12.plot("net12")
		# plt.show()

def test_performance(training_data,test_data,low,high,interval):
	results = []
	itrs = []
	etas = []
	rates = []
	for eta in np.arange(low,high,interval):
		itr = []
		rate = []
		print("========== {} ==========".format(eta))
		for i in range(3):
			net = min_max([2,10,2],eta)
			itr.append(net.learning(training_data))
			rate.append(net.evaluate(test_data))
		itr_avg = float(sum(itr)-max(itr)-min(itr))
		rate_avg = (sum(rate)-max(rate)-min(rate))
		itrs.append(itr_avg)
		etas.append(eta)
		rates.append(rate_avg)
		results.append((eta, itr_avg, rate_avg))
	plot(etas,itrs,rates)
		
	with open('results.txt','wb') as f:
		for eta, itr, rate in results:
			f.write('{}\t{:.2f}\t{:.2f}\n'.format(eta, itr, rate))

def plot(xs,y1s,y2s):
	plt.plot(xs,y1s)
	plt.plot(xs,y2s)
	#plt.show()
	plt.savefig('result.png')

if __name__ == '__main__':
	main()
