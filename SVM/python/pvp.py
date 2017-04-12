from svm import *
import svmutil
import numpy as np
import preprocess

problems = []
models = []

PS = 375 # the size of pieces during MIN-MAX

def main():
    # preprocess the train.txt, split it into 12 different files.
    preprocess.preprocess()
    get_problems()
    train_models()
    evaluate()

def get_problems():
    # first get 66 1v1 subproblems, then split them into several 
    # pieces of size PS, which would be used in later MIN-MAX.
    for i in range(12):
        yi, xi = svmutil.svm_read_problem('train_{}.txt'.format(i))
        if len(yi) <= PS:
            problems.append([(yi,xi)])
        else:
            each_problem = []
            cnt = len(yi)/PS
            for j in range(cnt):
                each_problem.append((yi[j*PS:(j+1)*PS],xi[j*PS:(j+1)*PS]))
            each_problem.append((yi[-cnt:],xi[-cnt:]))
            problems.append(each_problem)

def train_models():
    for i in range(11):
        for j in range(i+1, 12):
            each_models = []
            for yi,xi in problems[i]:
                each_model = []
                for yj,xj in problems[j]:
                    each_model.append(svmutil.svm_train(yi+yj,xi+xj,'-q -c 1024 -t 2 -g 16'))
                each_models.append(each_model)
            models.append((i,each_models))

def evaluate():
    target_y, target_x = svmutil.svm_read_problem('test.txt')
    num_err = 0
    num_correct = 0
    for i in range(len(target_y)):
        results = [0 for model in range(len(models))]
        x0, max_idx = gen_svm_nodearray(target_x[i])
        for j in range(len(models)):
            predict_label = minmax(j,x0)
            results[predict_label] = results[predict_label] + 1
        predict_y = np.argmax(results)
        if target_y[i] == predict_y:
            num_correct = num_correct + 1
        else:
            num_err = num_err + 1
    rate = (float)(num_correct) / (num_correct+num_err) * 100
    print("num_correct/Total: {}/{} ({:.2f}%)".format(num_correct,num_correct + num_err,rate))

# utils
def min(input_array,primary):
    for i in range(len(input_array)):
        if input_array[i] != primary:
            return input_array[i]
    return primary
def max(input_array,primary):
    for i in range(len(input_array)):
        if input_array[i] == primary:
            return primary
    return input_array[0]
def minmax(i,x):
    primary =  models[i][0]
    model = models[i][1]
    results = []
    for max_model in model:
        temp = [int(libsvm.svm_predict(min_model,x))  for min_model in max_model]
        results.append(min(temp,primary))
    result = max(results,primary)
    return result

if __name__ == "__main__":
    main()

     

