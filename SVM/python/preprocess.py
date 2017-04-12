import os

def preprocess():
    labels = [i for i in range(0,13)]
    for label in labels:
        filename = "train_{}.txt".format(label)
        if os.path.exists(filename):
            os.remove(filename)

    train = open("train.txt","r")

    for line in train:
        label = (line.split())[0]
        with open("train_{}.txt".format(label),"a") as f:
            f.write(line)

    train.close()