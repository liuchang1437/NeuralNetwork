train = open('../data/hw4-data.txt','r')
statistic = open('../data/statistic.txt','w')
data_processed = open('../data/data_processed.txt','w')
minimal = 100
maximal = -100
num = 0
for line in train:
    num = num + 1
    a1, a2 = line.strip().split()
    a1_base,a1_exp = a1.split('e')
    a1 = float(a1_base) * (10**(float(a1_exp)))
    a2_base,a2_exp = a2.split('e')
    a2 = float(a2_base) * (10**(float(a2_exp)))
    data_processed.write("{} {}\n".format(a1,a2))
    if min(a1,a2)< minimal:
        minimal = min(a1,a2)
    if max(a1,a2) > maximal:
        maximal = max(a1,a2)
statistic.write("minimal\tmaximal\tnum\n")
statistic.write("{}\t{}\t{}\n".format(minimal,maximal,num))
train.close()
statistic.close()
data_processed.close()
