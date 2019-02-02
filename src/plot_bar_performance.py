
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

data_names = ['HDFS','HBase']
indices = np.arange(len(data_names))
alpha = 0.15 # 1/x 

F1-HDFS = [0.63,0.71,0.74,0.62,0.64]

plt.figure(figsize=(8, 6))

plt.bar(indices, F1-HDFS[0], alpha, label="nb", color='y')
plt.bar(indices + alpha, f1_score1, alpha, label="f1-score",
         color='g')
plt.bar(indices + 2*alpha, g_measure1, alpha, label="g_measure", color='b')
plt.bar(indices + 3*alpha, auc1, alpha, label="auc", color='k')
plt.ylabel('score', fontsize =12)
plt.ylim(ymin=0,ymax=1)
plt.xticks(())
plt.legend(loc='best')
for i, c in zip(indices, clf_names):
    plt.text(i, -.05, c, fontsize = 12)


#plt.bar(indices, recall, alpha, label="recall", color='y')
#plt.bar(indices + alpha, f1_score, alpha, label="f1-score",
#         color='g')
#plt.bar(indices + 2*alpha, g_measure, alpha, label="g_measure", color='b')
#plt.bar(indices + 3*alpha, auc, alpha, label="auc", color='k')
#plt.ylabel('score', fontsize =12)
#plt.ylim(ymin=0,ymax=1)
#plt.xticks(())
#plt.legend(loc='best')
#for i, c in zip(indices, clf_names):
#    plt.text(i, -.05, c, fontsize = 12)


plt.savefig("../output/perf_hbase.eps")

plt.show()



