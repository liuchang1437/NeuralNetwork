# Neural Network
## Multilayer Quadratic Perceptron

+ It's an implementation of **MLQP**. The activation function is
  + $$y_i = sigmoid(\sum_i (u_{ji}y_i^2+v_{ji}y_i) +b_j)$$


### Running options

There are several functions it provides. You can change modes in `network.py`

```python
net = MLQP.MLQP([2,10,2])
# 1. batch mode
itr = net.SGD_batch(training_data, 10, 0.8) # (., epochs, batch size, learning rate)
# 2. on-line mode
itr = net.SGD_online(training_data, 0.8) # (., epochs, batch size, learning rate)
# verify the test_data 
result = net.evaluate(test_data)
# draw the classes distribution
net.plot()
# test the performance of different learning rate
test_performance(training_data, test_data,1.0,1.5,0.05)
```

+ Run it simply by `python network.py`
+ When test the performance of different learning rate, it executes 10 times each eta. And it will calculate the average value (not include the max and min ones).

