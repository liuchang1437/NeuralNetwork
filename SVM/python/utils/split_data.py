import os
os.chdir("../data")
labels = [i for i in range(0,13)]
for label in labels:
    filename = "label_{}.txt".format(label)
    if os.path.exists(filename):
        os.remove(filename)

train = open("train.txt","r")


lb = 0
count = 0
lb_count = []
for line in train:
    label = (line.split())[0]
    if int(label) == lb:
        count = count + 1
    else:
        lb_count.append((lb,count))
        lb = lb + 1
        count = 0
    with open("label_{}.txt".format(label),"a") as f:
        f.write(line)

with open("statistic.txt","w") as f:
    f.write("class\tcount\n")
    for label_name, label_count in lb_count:
        f.write("{}\t{}\n".format(label_name,label_count))
train.close()
