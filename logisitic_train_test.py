import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,\
                            hamming_loss, jaccard_similarity_score, matthews_corrcoef, \
                            zero_one_loss,roc_curve, auc,log_loss
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from time import time

def load_data(file_train,file_test,file_fea):
    data_train = pd.read_csv(file_train)
    data_test = pd.read_csv(file_test)
    with open(file_fea, 'rb') as f:
        selected_feat_names = pickle.load(f)
    print(selected_feat_names)
    x_train = data_train[selected_feat_names]
    y_train = data_train['label']
    x_test = data_test[selected_feat_names]
    y_test = data_test['label']
    return x_train,y_train,x_test,y_test

def classify(x_train,y_train,x_test):

    X_train, X_test, y_train = x_train,x_test,y_train
    LR = LogisticRegression(solver='saga',max_iter=1000,C=0.8,tol=0.01)
    t0 = time()
    LR.fit(X_train, y_train)
    tt = time() - t0
    print("Los in {} seconds".format(round(tt, 3)))
    y_predict = LR.predict(X_test)
    joblib.dump(LR, r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\lr_model_params.pkl')
    return y_predict,LR



def calculate_metrics(y_test, y_predict,y_score):

    acc = accuracy_score(y_test, y_predict)
    print("\n\nAccuracy {} %".format(round(acc * 100, 3)))
    confusion = confusion_matrix(y_test, y_predict)
    print("\n\nConfusion Matrix: \n\n {}".format(confusion))
    print("The precision,recall, f-score obtained for this model are")
    classific_report = classification_report(y_test, y_predict)
    print("\n\nClassification Scores: \n\n {}".format(classific_report))
    hamm_loss = hamming_loss(y_test, y_predict)
    print("\n\nHamming Loss {}".format(hamm_loss))
    jaccard_similarity = jaccard_similarity_score(y_test, y_predict)
    print("\n\nJaccard Similarity Score {}".format(jaccard_similarity))
    matthews_corr = matthews_corrcoef(y_test, y_predict)
    print("\n\nMatthews corrcoef {}".format(matthews_corr))
    zero_one_los = zero_one_loss(y_test, y_predict)
    print("\n\nZero-One Loss {}".format(zero_one_los))
    log_los = log_loss(y_test,y_score)#每个样本的 log loss 是给定的分类器的 negative log-likelihood 真正的标签
    print("\n\nlog_Loss {}".format(log_los))

def ROC_curve(X_test,y_test,LR):

    y_score = LR.predict_proba(X_test)
    y_test = [i for i in y_test.values]
    y_test.extend([2])
    y_test = label_binarize(y_test, classes=[0, 1])
    y_test = np.delete(y_test, -1, axis=0)
    samples = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(samples):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[1], tpr[1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_auc[1] = auc(fpr[1], tpr[1])
    print('The roc rate for 1st attribute is ', roc_auc[1])
    plt.show()


def main():

    file_train = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_train_new_norm.csv'
    file_test = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_test_new_norm.csv'
    file_fea = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\selected_feat_names.pkl'
    x_train, y_train, x_test, y_test = load_data(file_train,file_test,file_fea)
    y_predict , LR= classify(x_train,y_train,x_test)
    y_score = LR.predict_proba(x_test)
    # print(LR.coef_)#权重向量
    calculate_metrics(y_test, y_predict,y_score)
    ROC_curve(x_test, y_test,LR)
    lr_w = joblib.load(r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\lr_model_params.pkl')
    print(lr_w)#模型参数

if __name__ == "__main__":
    main()