# -*- coding: utf-8 -*
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import pickle
import numpy as np
from logisitic_train_test import load_data
file_train = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_train_new_norm.csv'
file_test = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_test_new_norm.csv'
file_fea = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\selected_feat_names.pkl'
x_train, y_train, x_test, y_test = load_data(file_train,file_test,file_fea)
lr = LogisticRegressionCV(cv=5, random_state=0,solver='saga').fit(x_train, y_train)#可以自动对正则化参数c交叉验证，默认cs=10，有10个选择
y_predict = lr.predict(x_test)
print(lr.score(x_train,y_train))
