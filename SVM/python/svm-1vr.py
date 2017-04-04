import svmutil
import numpy as np
from svm import *
import ctypes

problems = []
models = []

problems_sgl_tp = []
for i in range(12): # 0,...,10
    yi, xi = svmutil.svm_read_problem('data/label_{}.txt'.format(i))
    problems_sgl_tp.append([yi,xi])

for i in range(12):
    yi = problems_sgl_tp[i][0]
    yi = [1 for yyi in yi]
    xi = problems_sgl_tp[i][1]
    for j in range(12):
        if i!=j:
            yj = problems_sgl_tp[j][0]
            yj = [-1 for yyj in yj]
            xj = problems_sgl_tp[j][1]
            yi = yi + yj
            xi = xi + xj
    problems.append([yi,xi])

for y,x in problems:
    models.append(svmutil.svm_train(y,x,'-c 1024 -t 2 -g 32'))

t_y, t_x = svmutil.svm_read_problem('data/test.txt')
size = len(t_y)
err = 0
correct = 0
for i in range(size):
    result = []
    x0, max_idx = gen_svm_nodearray(t_x[i])
    for j in range(12):
        d_ptr = ctypes.c_double()
        # d_ptr = [0 for _ in range(66)]
        # d_ptr = (ctypes.c_double * len(d_ptr))(*d_ptr)
        libsvm.svm_predict_values(models[j], x0, d_ptr)
        # print("=========================")
        # print(d_ptr.value)
        # print("=========================")
        result.append(d_ptr.value)
    p_y = np.argmax(result)
    if t_y[i] == p_y:
        correct = correct + 1
    else:
        err = err + 1
rate = (float)(correct) / (correct+err) * 100
print("Correct/Total: {}/{} ({:.2f}%)".format(correct,correct + err,rate))

     

