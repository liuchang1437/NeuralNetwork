import numpy as np

def load_data():
    training_data = []
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
        y = vectorized_result(int(y))
        training_data.append((x,y))
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
    return training_data, test_data

def vectorized_result(j):
    e = np.zeros((2,1))
    e[j] = 1.0
    return e