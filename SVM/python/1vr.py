from svm import *
import svmutil
import numpy as np
import preprocess
import ctypes

problems = []
models = []

def main():
    # preprocess the train.txt, split it into 12 different files.
    preprocess.preprocess()
    get_problems()
    train_models()
    evaluate()

def get_problems():
    # get subproblems: 0-rest, 1-rest, ...
    each_problem = []
    for i in range(12):
        yi, xi = svmutil.svm_read_problem('train_{}.txt'.format(i))
        each_problem.append([yi,xi])
    for i in range(12):
        yi = each_problem[i][0]
        yi = [1 for label in yi]
        xi = each_problem[i][1]
        for j in range(12):
            if i!=j:
                yj = each_problem[j][0]
                yj = [-1 for label in yj]
                xj = each_problem[j][1]
                yi = yi + yj
                xi = xi + xj
        problems.append([yi,xi])

def train_models():
    # train all the 12 subproblems
    for y,x in problems:
        models.append(svmutil.svm_train(y,x,'-q -c 1024 -t 2 -g 32'))

def evaluate():
    target_y, target_x = svmutil.svm_read_problem('test.txt')
    num_err = 0
    num_correct = 0
    for i in range(len(target_y)):
        results = []
        x0, max_idx = gen_svm_nodearray(target_x[i])
        for j in range(12):
            c_double_ptr = ctypes.c_double()
            libsvm.svm_predict_values(models[j], x0, c_double_ptr)
            results.append(c_double_ptr.value)
        predict_y = np.argmax(results)
        if target_y[i] == predict_y:
            num_correct = num_correct + 1
        else:
            num_err = num_err + 1
    rate = (float)(num_correct) / (num_correct+num_err) * 100
    print("num_correct/Total: {}/{} ({:.2f}%)".format(num_correct,num_correct + num_err,rate))

if __name__ == "__main__":
    main()
     

