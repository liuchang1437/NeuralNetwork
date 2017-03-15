# Neural Network
## Perceptron

+ It's implement a Perceptron. The learning rate is :
  + $$\bm{W}^{new} = \bm{W}^{old} + \alpha \bm{e}\bm{p}^T$$
  + $$\bm{b}^{new} = \bm{b}^{old} + \alpha \bm{e}\b$$
+ The program could draw the classes distribution after finishing the learning.
+ The program can also learning at different learning rates to compare within them.
+ Run it simply by `python perceptron.py`

## Multilayer Quadratic Perceptron

+ It's an implementation of **MLQP**. The activation function is
  + $$y_i = sigmoid(\sum_i (u_{ji}y_i^2+v_{ji}y_i) +b_j)$$
+ The program could draw the classes distribution after finishing the learning.
+ The program can also learning at different learning rates to compare within them.


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


## MIN-MAX

For each classes, the program randomly split it into two samples of the same size, say they are *t11, t12, t01 and t02.* By merging them together, we get $4$ new sub-training sets: *t11 + t01, t11 + t02, t12 + t01, t12 + t02*. For each sub-training sets, we use it to train a MLQP network. By using MIN-MAX rules, we can get the final results. 