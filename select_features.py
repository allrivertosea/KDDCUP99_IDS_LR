import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from data_processing import *
def select_feature(data_train_new):
    #随机森林特征选择，不需要进行标准化，归一化即可进行
    X = data_train_new.iloc[:,1:115]
    y = data_train_new.iloc[:,115]
    selected_feat_names = set()
    rfc = RandomForestClassifier(n_jobs=-1)#用于拟合和预测的并行运行的工作（作业）数量。如果值为-1，那么工作数量被设置为核的数量。
    rfc.fit(X, y)
    print("training finished")
    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]# descending order
    for f in range(X.shape[1]):
        if f < 50:
            selected_feat_names.add(X.columns[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))
    for i in range(9):
        tmp = set()
        rfc = RandomForestClassifier(n_jobs=-1)
        rfc.fit(X, y)
        print("training finished")
        importances = rfc.feature_importances_
        indices = np.argsort(importances)[::-1] # descending order
        for f in range(X.shape[1]):
            if f < 50: # need roughly more than 40 features according to experiments
                tmp.add(X.columns[indices[f]])
            print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))
        selected_feat_names &= tmp
    with open(r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\selected_feat_names.pkl', 'wb') as f:
        pickle.dump(list(selected_feat_names), f)
    selected_feat_names = list(selected_feat_names)
    return selected_feat_names

if __name__ == "__main__":
    file_train = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_train_new_norm.csv'
    data_train_new = pd.read_csv(file_train)
    select_feature(data_train_new)