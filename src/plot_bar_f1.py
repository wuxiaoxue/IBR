
import matplotlib.pyplot as plt
import pandas
import numpy as np
#
#clf_names =['nb','lr','svm','mlp','rf']
#data_set = '../output/hdfs_dmr_output.csv'
#data_set1 = '../output/hbase_dmr_output.csv'
#
#col_names = ['clf_name','TP', 'FN', 'TN', 'FP', 'pd', 'pf', 'prec', 'f1', 'g1', 'sr', 'auc', 'PofB20', 'opt']
#data = pandas.read_csv(data_set, names=col_names, header=None)
#clf_name = data.clf_name
#recall = data.pd
#f1_score = data.f1
#g_measure = data.g1
#auc = data.auc
#
#data1 = pandas.read_csv(data_set1, names=col_names, header=None)
#clf_name1 = data1.clf_name
#recall1 = data1.pd
#f1_score1 = data1.f1
#g_measure1 = data1.g1
#auc1 = data1.auc

data_names =['HDFS','HBase']
indices = np.arange(len(data_names))
alpha = 0.15 # 1/x 

data1=[0.76,0.63]
data2=[0.83,0.71]
data3=[0.85,0.74]
data4=[0.75,0.62]
data5=[0.62,0.64]

#F1-HDFS = [0.63,0.71,0.74,0.62,0.64]
#F1-HBase = [0.76,0.83,0.85,0.75,0.62]

plt.figure(figsize=(8, 6))

plt.bar(indices, data1, alpha, label="NB", color='red')
plt.bar(indices + alpha, data2, alpha, label="LR",
         color='yellow')
plt.bar(indices + 2*alpha, data3, alpha, label="SVM", color='green')
plt.bar(indices + 3*alpha, data4, alpha, label="MLP", color='blue')
plt.bar(indices + 4*alpha, data5, alpha, label="RF", color='purple')
plt.ylabel('F1-score', fontsize =12)
plt.ylim(ymin=0,ymax=1)
plt.xticks(())
plt.legend(loc='best')
for i, c in zip(indices, data_names):
    plt.text(i+alpha, -.05, c, fontsize = 12)

plt.savefig("../output/bar-f1.eps")

plt.show()



