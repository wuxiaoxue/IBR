# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:04:03 2019

@author: Administrator
"""
#total IBR	Predicted PBRs	has components	datanode	namenode	hdfs-client	others
#HDFS-invalid	800	99	54	14	17	9	14
#Hbase-invalid							

import matplotlib.pyplot as plt

datas =[19,23,10,4,12]
colors =['blue','red','lightgreen','yellow','gray']
labels=['datanode','namenode','hdfs-client','web-hdfs','all others']
explode =[0,0.2,0,0,0]
plt.pie(datas,colors=colors,labels = labels, explode=explode,autopct = '%1.1f%%')
plt.savefig("../output_v2/pie-hdfs.eps")
plt.axis("equal")
plt.show()