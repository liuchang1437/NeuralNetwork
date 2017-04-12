from svm import *
import svmutil
import numpy as np
import preprocess

problems = []
models = []

def main():
    # preprocess the train.txt, split it into 12 different files.
    preprocess.preprocess()
    get_problems()
    train_models()
    evaluate()

def get_problems():
    # get 66 subproblems: 0-1, 0-2, ... 
    for i in range(11):
        for j in range(i+1, 12):
            yi, xi = svmutil.svm_read_problem('train_{}.txt'.format(i))
            yj, xj = svmutil.svm_read_problem('train_{}.txt'.format(j))
            problems.append([yi+yj,xi+xj])

def train_models():
    # train all the 66 subproblems
    for y,x in problems:
        models.append(svmutil.svm_train(y,x,'-q -c 64 -t 2 -g 32'))

def evaluate():
    target_y, target_x = svmutil.svm_read_problem('test.txt')
    num_err = 0
    num_correct = 0
    for i in range(len(target_y)):
        results = [0 for j in range(12)]
        x0, max_idx = gen_svm_nodearray(target_x[i])
        for model in models:
            predict_label = int(libsvm.svm_predict(model, x0))
            results[predict_label] = results[predict_label] + 1
        predict_y = np.argmax(results)
        if target_y[i] == predict_y:
            num_correct = num_correct + 1
        else:
            num_err = num_err + 1
    rate = (float)(num_correct) / (num_correct+num_err) * 100
    print("num_correct/Total: {}/{} ({:.2f}%)".format(num_correct,num_correct+num_err,rate))


if __name__ == "__main__":
    main()

     

