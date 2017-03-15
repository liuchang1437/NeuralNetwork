import random
import numpy as np

def random_load_data():
    training_y1 = []
    training_y0 = []
    test_data = []
    train_txt = open("../data/two_spiral_train.txt", "r")
    test_txt = open("../data/two_spiral_test.txt", "r")
    # the training_data is in the format as bellow: 
    # ([x1,
    #  x2],[y1.
    #       y2])
    for line in train_txt:
        x0, x1, y = line.split()
        x = np.array([float(x0), float(x1)]).reshape(2, 1)
        if int(y) == 0:
            y = vectorized_result(int(y))
            training_y0.append((x,y))
        else:
            y = vectorized_result(int(y))
            training_y1.append((x,y))
    # the test_data is in the format as bellow: 
    # ([x1,
    #  x2],y)
    for line in test_txt:
        x0, x1, y = line.split()
        x = np.array([float(x0), float(x1)]).reshape(2, 1)
        y = int(y)
        test_data.append((x,y))
    train_txt.close()
    test_txt.close()

    random.shuffle(training_y0)
    random.shuffle(training_y1)

    training_11 = training_y1[:len(training_y1)/2]
    training_12 = training_y1[len(training_y1)/2:]
    training_01 = training_y0[:len(training_y0)/2]
    training_02 = training_y0[len(training_y0)/2:]

    t1 = training_11+training_01
    t2 = training_11+training_02
    t3 = training_01+training_12
    t4 = training_02+training_12

    # random.shuffle(t1)
    # random.shuffle(t2)
    # random.shuffle(t3)
    # random.shuffle(t4)

    return (t1, t2, t3, t4), test_data

def vectorized_result(j):
    e = np.zeros((2,1))
    e[j] = 1.0
    return e