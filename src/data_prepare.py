import re
import csv
import codecs
import pandas
import numpy as np
from sklearn.utils import shuffle
from Tools.scripts.objgraph import ignore
from scipy import sparse


data_name = "hbase-invalid"
#input_file0 = "../datasource/" + data_name + "_0.csv"
#input_file1 = "../datasource/" + data_name + "_1.csv"

input_file0 = "../datasource/hbase-invalid-part1.csv"
input_file1 = "../datasource/hbase-invalid-part2.csv"

col_names = ['content','label']
sourcedata0 = pandas.read_csv(input_file0, names=col_names).fillna('')
#sourcedata0 = sourcedata0[:4000]  # for large sized imbalance data
sourcedata1 = pandas.read_csv(input_file1, names=col_names).fillna('')
#sourcedata2 = shuffle(sourcedata2)

data = np.vstack((sourcedata0, sourcedata1))
data = shuffle(data)
data_content = data[:,0]
data_label = data[:,1]

out = open("../input/" + data_name + "_labeled.csv", "w", newline='')
writer = csv.writer(out, delimiter=',')
for item in range(0, len(data_label)-1):
    row = [data_content[item], data_label[item]]
#    if not(regexp.search(sourcedata_content[item])):
    writer.writerow(row)
out.close()

#out = open("../datasource/1_6_3/" + data_name + "_4k__cand.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#for item in range(0, len(cand_label)-1):
#    row = [cand_content[item], cand_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()
#
#out = open("../datasource/1_6_3/" + data_name + "_4k_test.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#for item in range(0, len(test_label)-1):
#    row = [test_content[item], test_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()


#sourcedata_cand_content = sourcedata_cand[:, 0].tolist()
#sourcedata_cand_label = sourcedata_cand[:, 1].tolist()
#
#sourcedata_test_content = sourcedata_test[:, 0].tolist()
#sourcedata_test_label = sourcedata_test[:, 1].tolist()
#
#print('******train.shape*******\n' + str(sourcedata_train.shape))
#print('******cand.shape*******\n' + str(sourcedata_cand.shape))
#print('******test.shape*******\n' + str(sourcedata_test.shape))
#
#
#out = open("../datasource/cross_train/cve5000.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#
#
#for item in range(0, len(sourcedata_train_label)-1):
#    row = [sourcedata_train_content[item], sourcedata_train_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()
#
#out = open("../datasource/cross_train/openstack_cand.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#for item in range(0, len(sourcedata_cand_label)-1):
#    row = [sourcedata_cand_content[item], sourcedata_cand_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()
#
#out = open("../datasource/cross_train/openstack_test.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#for item in range(0, len(sourcedata_test_label)-1):
#    row = [sourcedata_test_content[item], sourcedata_test_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()


#
#def read_csv_row(f, r):
#    with open(f, 'rb') as csvfile:
#        R = [row[r] for row in csv.DictReader(csvfile)]
#    return R
#
#csv_file0 = csv.reader(open(input_file0, 'r', errors="ignore"))
##csv_file1 = csv.reader(open(input_file1, 'r', errors ="ignore"))
#
#sourcedata0 = pandas.read_csv(input_file0, names=col_names)
#
#data0 = []
#data1 = []
#
#for item in csv_file1:
#    data1.append(item)
#    
#for item in csv_file0:
#    data0.append(item)
#
#for item in sourcedata:
#	if item[1] == "0":
#		data0.append(item)
#	elif item[1] == "1":
#		data1.append(item)
##num = 400 - len(data1)
#
#print(data0)
##print(data1)
#sourcedata = np.vstack((data1,data0))
#sourcedata = shuffle(sourcedata)
#print(sourcedata.shape)
##
#out = open("../input/"+data_name+"_train.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#
#for item in range(0, int((len(sourcedata_label)-1)*1/2)):
#    row = [sourcedata_content[item], sourcedata_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()
#
#out = open("../input/"+data_name+"_test.csv", "w", newline='')
#writer = csv.writer(out, delimiter=',')
#
#for item in range(int((len(sourcedata_label)-1)*1/2)+1, len(sourcedata_label)-1):
#    row = [sourcedata_content[item], sourcedata_label[item]]
##    if not(regexp.search(sourcedata_content[item])):
#    writer.writerow(row)
#out.close()

print("wonderful")

