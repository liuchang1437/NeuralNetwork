import svmutil
import numpy as np
from svm import *
import ctypes

problems = []
models = []

for i in range(11): # 0,...,10
    for j in range(i+1, 12): # (i,i+1),...,(i,11)
        yi, xi = svmutil.svm_read_problem('data/label_{}.txt'.format(i))
        yj, xj = svmutil.svm_read_problem('data/label_{}.txt'.format(j))
        y = yi+yj
        x = xi+xj
        problems.append([y,x])

for y,x in problems:
    models.append(svmutil.svm_train(y,x,'-c 1024 -t 2 -g 32'))

t_y, t_x = svmutil.svm_read_problem('data/test.txt')
size = len(t_y)
err = 0
correct = 0
for i in range(size):
    result = [0 for j in range(12)]
    x0, max_idx = gen_svm_nodearray(t_x[i])
    for model in models:
        predict_label = int(libsvm.svm_predict(model, x0))
        result[predict_label] = result[predict_label] + 1
        # ddd = ctypes.c_double()
        # # >>> nums = [1, 2, 3]
        # # >>> a = (ctypes.c_double * len(nums))(*nums)
        # libsvm.svm_predict_values(model, x0,ddd)
        # print("=========================")
        # print(ddd)
        # print("=========================")
    p_y = np.argmax(result)
    if t_y[i] == p_y:
        correct = correct + 1
    else:
        err = err + 1
rate = (float)(correct) / (correct+err) * 100
print("Correct/Err: {}/{} ({:.2f}%)".format(correct,err,rate))

     

