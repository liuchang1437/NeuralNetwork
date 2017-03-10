import loader
import MLQP
import numpy as np

def main():
	training_data, test_data = loader.load_data()
	# the network consists of 3 layers, each of which contains [2, 10, 2] neurons.
	net = MLQP.MLQP([2,10,2])
	# itr = net.SGD_batch(training_data, 10, 0.8) # (., epochs, batch size, learning rate)
	itr = net.SGD_online(training_data, 0.8) # (., epochs, batch size, learning rate)
	result = net.evaluate(test_data)
	net.plot()

	#test_performance(training_data, test_data,1.0,1.5,0.05)

def test_performance(training_data,test_data,low,high,interval):
	results = []
	for eta in np.arange(low,high,interval):
		itr = []
		rate = []
		print("========== {} ==========".format(eta))
		for i in range(10):
			net = MLQP.MLQP([2,10,2])
			itr.append(net.SGD_online(training_data,eta))
			rate.append(net.evaluate(test_data))
		itr_avg = float(sum(itr)-max(itr)-min(itr))/8
		rate_avg = (sum(rate)-max(rate)-min(rate))/8
		results.append((eta, itr_avg, rate_avg))
		
	with open('results.txt','wb') as f:
		for eta, itr, rate in results:
			f.write('{}\t{:.2f}\t{:.2f}\n'.format(eta, itr, rate))

if __name__ == '__main__':
	main()
