import loader
import MLQP

def main():
	training_data, test_data = loader.load_data()
	# the network consists of 3 layers, each of which contains [2, 10, 2] neurons.
	net = MLQP.MLQP([2,10,2])
	#net.SGD_batch(training_data, 10, 0.8) # (., epochs, batch size, learning rate)
	net.SGD_online(training_data, 0.8) # (., epochs, batch size, learning rate)
	net.evaluate(test_data)

if __name__ == '__main__':
	main()
