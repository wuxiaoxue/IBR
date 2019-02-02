# coding: utf-8
import csv
import time

import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

import dimension_reduce as dr
import model_measure_functions as mf
#from smote import get_smote_borderline_1
from smote import get_smote_standard
from smote import get_smoteenn

data_name = 'hdfs'  # the csv file name of input folder
clf_name = 'rf'
#clf_name = 'nb'  # classifier: can be 'lr', 'rf', 'mlp', 'svm'

data_set = '../input/' + data_name + '.csv'
col_names = ['content', 'label', 'proba']
data = pandas.read_csv(data_set, names=col_names, header=None).fillna('')
output = '../output/ES_' + data_name+'_' + clf_name+'_output.csv'
csv_file = open(output, "w", newline='')
writer = csv.writer(csv_file, delimiter=',')

i=1
while i<51: 
    data = shuffle(data)
    data_content = data.content
    data_label = data.label.tolist()
    
    count_vect = CountVectorizer(stop_words='english')
    data_content_matrix = count_vect.fit_transform(data_content)
    data_content_matrix_dmr = dr.selectFromLinearSVC(data_content_matrix,data_label)
#    data_content_matrix_dmr_smt,data_label_smt = get_smoteenn(data_content_matrix_dmr,data_label)

    matrix_content = data_content_matrix_dmr
    
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

    predicted = cross_val_predict(clf, matrix_content, data_label, cv=10)
    predicted_proba = cross_val_predict(clf, matrix_content, data_label, cv=10,
                                        method="predict_proba")
    #     print(predicted,predicted_proba)
    TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate, auc, PofB20, opt \
        = mf.model_measure_with_cross(predicted, predicted_proba, data_label)
#    print(i,TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate, auc, PofB20, opt)
    print(i)
    writer.writerow([i,TP, FN, TN, FP, pd, pf, prec, f_measure,
                     g_measure, success_rate, auc, PofB20, opt])
    i = i+1
csv_file.close()
print("****************finished**********************")
#print(str(end - start))