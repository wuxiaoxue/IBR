# -*- coding: utf-8 -*-

# Non-interactive logisticRegression Text Classification
# By: Xiaoxue Wu, 9/08/2018

import csv
import time
import pandas
import numpy as np

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import dimension_reduce as dr

from sklearn.utils import shuffle

data_name = 'hdfs'
clf_names = ['svm'] #
train_data_set = '../input/hdfs-labeled.csv'
target_data_set = '../input/hdfs-invalid-latest.csv'
target_data_set2 = '../input/hbase-invalid.csv'

col_names1 = ['content', 'label']
train_data = pandas.read_csv(train_data_set, names=col_names1, header=None).fillna('')
col_names2 =['content']
target_data = pandas.read_csv(target_data_set, names=col_names2).fillna('')
#target_data = shuffle(target_data)
train_content = train_data.content
train_label = train_data.label.tolist()

target_comment = target_data.content

vectorizer = CountVectorizer(stop_words='english')
train_content_matrix = vectorizer.fit_transform(train_content)
target_comment_matrix = vectorizer.transform(target_comment)

train_content_matrix_dmr, target_comment_matrix_dmr \
    = dr.selectFromLinearSVC2(train_content_matrix,train_label,target_comment_matrix) 

output = '../output_v2/invalid_' + data_name + '_svm_output_latest.csv'    
csv_file = open(output, "w", newline='')
writer = csv.writer(csv_file, delimiter=',')

for clf_name in clf_names:
    if clf_name == 'lr':
        clf = LogisticRegression()
    elif clf_name == 'svm':
        # the kernel can also be 'linear', 'rbf','polynomial','sigmoid', etc.
        clf = svm.SVC(kernel='linear', probability=True)
    elif clf_name == 'mlp':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
    elif clf_name == 'nb':
        clf = MultinomialNB()
    elif clf_name == 'rf':
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
    else:
        print('分类器名称仅为\'lr,svm,mlp,nb,rf\'中的一种')
   
    clf.fit(train_content_matrix_dmr,train_label)
    predicted = clf.predict(target_comment_matrix_dmr)   
    
    for item in predicted:
        writer.writerow([item])
#    print(predicted)

#    num = 0
#    for item in predicted:
#        if item ==1:
#            num = num + 1
#    writer.writerow([clf_name, num])
#    print(num)
    
csv_file.close()
#    print(output + '**************** finished************************')
print("Great! All Finished")

