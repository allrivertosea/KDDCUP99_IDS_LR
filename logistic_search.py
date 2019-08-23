# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import pickle
from data_processing import *
from scoring import cost_based_scoring
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def load_data(file_train,file_fea):
    data_train = pd.read_csv(file_train)
    with open(file_fea, 'rb') as f:
        selected_feat_names = pickle.load(f)
    print(selected_feat_names)
    x_train = data_train[selected_feat_names]
    y_train = data_train['label']
    return x_train,y_train

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
def optimize_params(x_train,y_train):
    y = y_train
    X = x_train
    LR = LogisticRegression(solver='saga',max_iter=1000)
    param_grid = {"tol": [1e-4, 1e-3, 1e-2], "C": [0.4, 0.6, 0.8]}
    grid_search = GridSearchCV(LR,param_grid,scoring='f1', cv=3)
    grid_search.fit(X, y)
    report(grid_search.cv_results_)
    print("optimization params:",grid_search.best_params_)#得到最优参数
    print("grid search finished")

if __name__ == "__main__":
    file_train = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_train_new_norm.csv'
    file_fea = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\selected_feat_names.pkl'
    x_train, y_train = load_data(file_train,file_fea)
    optimize_params(x_train,y_train)
#n_estimators=75, learning_rate=1.5