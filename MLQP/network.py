import loader
import MLQP
import numpy as np

def main():
	training_data, test_data = loader.load_data()
	# the network consists of 3 layers, each of which contains [2, 10, 2] neurons.
	# net = MLQP.MLQP([2,10,2])
	# net.SGD_batch(training_data, 10, 0.8) # (., epochs, batch size, learning rate)
	# itr = net.SGD_online(training_data, 0.8) # (., epochs, batch size, learning rate)
	# result = net.evaluate(test_data)
	# #net.plot()

	#test_performance(training_data, test_data)
	test_tens(training_data,test_data)

def test_tens(training_data, test_data):
	eta = 0.95
	itr = 0.0
	rate = 0.0
	print("========== {} ==========".format(eta))
	for i in range(10):
		net = MLQP.MLQP([2,10,2])
		itr = itr + net.SGD_online(training_data,eta)
		rate = rate + net.evaluate(test_data)
	print(('{}\t{:.2f}\t{:.2f}'.format(eta, itr/10, rate/10)))

def test_performance(training_data,test_data):
	results = []
	for eta in np.arange(0.4,1.0,0.05):
		itr = 0.0
		rate = 0.0
		print("========== {} ==========".format(eta))
		for i in range(10):
			net = MLQP.MLQP([2,10,2])
			itr = itr + net.SGD_online(training_data,eta)
			rate = rate + net.evaluate(test_data)
		results.append((eta, itr/10, rate/10))
		
	with open('results_04_1.txt','wb') as f:
		for eta, itr, rate in results:
			f.write('{}\t{:.2f}\t{:.2f}\n'.format(eta, itr, rate))

if __name__ == '__main__':
	main()
