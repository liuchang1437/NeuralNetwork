import svmutil
import numpy as np
from svm import *
import ctypes

def min(rs,criteria):
    for i in range(len(rs)):
        if rs[i] != criteria:
            return rs[i]
    return criteria
def max(rs,criteria):
    for i in range(len(rs)):
        if rs[i] == criteria:
            return criteria
    return rs[0]

K = 375

problems = []

for i in range(12): # 0,...,10
    yi, xi = svmutil.svm_read_problem('data/label_{}.txt'.format(i))
    if len(yi) <= K:
        problems.append([(yi,xi)])
    else:
        problem_each = []
        cnt = len(yi)/K
        for j in range(cnt):
            problem_each.append((yi[j*K:(j+1)*K],xi[j*K:(j+1)*K]))
        #problem_each.append((yi[cnt*K:],xi[cnt*K:]))
        problem_each.append((yi[-cnt:],xi[-cnt:]))
        problems.append(problem_each)

# ------------ The format of problems --------------
# problems = [
#     [(y1,x1), (y2,x2),..], where the length of each yi <= K
#     ...
#     [..]
# ]

models = []

for i in range(11):
    for j in range(i+1, 12):
        temp_total = []
        for yi,xi in problems[i]:
            temp = []
            for yj,xj in problems[j]:
                temp.append(svmutil.svm_train(yi+yj,xi+xj,'-c 1024 -t 2 -g 16svm'))
            temp_total.append(temp)
        models.append((i,temp_total))

def minmax(i,x):
    primary =  models[i][0]
    model = models[i][1]
    max_results = []
    for tomax in model:
        min_results = [int(libsvm.svm_predict(tomin,x))  for tomin in tomax]
        #print(min_results)
        max_results.append(min(min_results,primary))
    max_result = max(max_results,primary)
    return max_result



t_y, t_x = svmutil.svm_read_problem('data/test.txt')
size = len(t_y)
err = 0
correct = 0
for i in range(size):
    result = [0 for zzz in range(len(models))]
    x0, max_idx = gen_svm_nodearray(t_x[i])
    for j in range(len(models)):
        predict_label = minmax(j,x0)
        result[predict_label] = result[predict_label] + 1
    p_y = np.argmax(result)
    if t_y[i] == p_y:
        correct = correct + 1
    else:
        err = err + 1
rate = (float)(correct) / (correct+err) * 100
print("Correct/Total: {}/{} ({:.2f}%)".format(correct,correct + err,rate))

     

