# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:47:38 2019

@author: KAT
Input: X_train, y_train, X_test, y_test (Oversampling.py)
Output: plots and csv tables with performance of different machine learning models
Purpose: evaluate how different machine learning models perform on the training dataset (classify disappeared companies)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict

score={'AUC':'roc_auc', 
           'RECALL':'recall',
           'PRECISION':'precision',
           'F1':'f1'}

X_train = pd.read_csv(r"disap\X_all_train_wo_os.csv", low_memory=False, index_col = 0).iloc[:, 2:]
y_train = pd.read_csv(r"disap\y_train_wo_os.csv", low_memory=False, index_col = 0)["bnkrpt"].squeeze()
X_test = pd.read_csv(r"disap\X_all_test.csv", low_memory=False, index_col = 0).iloc[:, 2:]
y_test = pd.read_csv(r"disap\y_test.csv", low_memory=False, index_col = 0)["bnkrpt"].squeeze()
dataname = "disap"

LogReg = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', LogisticRegression(solver='lbfgs', random_state=0))
            ])
LogReg_para = {}
RandF = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', RandomForestClassifier(random_state=0))
            ])
RandF_para = {'classification__n_estimators':[20, 200, 800], 'classification__max_depth':[2, 10, 20]}
AdaBoost = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', AdaBoostClassifier(random_state=0))
            ])
AdaBoost_para = {'classification__n_estimators':[20, 50, 100, 200, 400, 800]}
SVM = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', SVC(decision_function_shape='ovr', degree=3, gamma='auto'))
            ]) 
SVM_para = {'classification__C':[0.01, 0.1, 1, 10], 'classification__kernel':('linear', 'rbf')}
NaivBay = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', GaussianNB())
            ])
NaivBay_para = {}
Knn = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski'))
            ])
Knn_para = {'classification__n_neighbors': (10, 25)}
DecTree = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', DecisionTreeClassifier())
            ])
DecTree_para = {}

#clasifier_names = ["Logistic Regression", "Random Forest", "Adaptive Boosting", "Support Vector Machines", "Naive Bayes", "K Nearest Neighbours"]
#classifiers = [LogReg, RandF, AdaBoost, SVM, NaivBay, Knn]
#parameters = [LogReg_para, RandF_para, AdaBoost_para, SVM_para, NaivBay_para, Knn_para]

clasifier_names = ["Logistic Regression", "Random Forest", "Adaptive Boosting", "Naive Bayes", "K-nearest neighbours", "Decision Tree"]
classifiers = [LogReg, RandF, AdaBoost, NaivBay, Knn, DecTree]
parameters = [LogReg_para, RandF_para, AdaBoost_para, NaivBay_para, Knn_para, DecTree_para]


mean_res = list()

for i in range(len(classifiers)):
    clf = GridSearchCV(classifiers[i], parameters[i], cv=5, scoring=score, n_jobs=-1, refit=False, return_train_score=True)
    clf.fit(X_train, y_train)
    results2 = list()
    for j in range(len(clf.cv_results_["params"])):
        param_result = [clasifier_names[i]]
        for key in clf.cv_results_.keys():
            param_result.extend([clf.cv_results_[key][j]])
        results2.append(param_result)
        mean_res.append([clasifier_names[i], clf.cv_results_["params"][j], clf.cv_results_["mean_test_F1"][j], clf.cv_results_["mean_test_RECALL"][j], clf.cv_results_["mean_test_PRECISION"][j], clf.cv_results_["mean_train_F1"][j], clf.cv_results_["mean_train_RECALL"][j], clf.cv_results_["mean_train_PRECISION"][j]])
    results2 = pd.DataFrame(results2, columns = ["algorithm"] + list(clf.cv_results_.keys()))
    results2.to_csv("results_ML_{}_{}.csv".format(clasifier_names[i], dataname), index=False)
    print(clasifier_names[i])
    print(clf.cv_results_)

colors = {"Logistic Regression":"red", "Random Forest":"blue", "Adaptive Boosting":"green", "Naive Bayes":"orange", "Support Vector Machines":"black", "K-nearest neighbours":"purple", "Decision Tree":"grey"}

matplotlib.rcParams.update({'font.size': 22})

#plot mean test scores
fig, ax = plt.subplots(figsize=(15,15))
ax.plot([], [], ' ', label="Top algorithm is annotated with F1 score")
for i in range(len(mean_res)):
    x = mean_res[i][3]
    y = mean_res[i][4]
    ax.scatter(x, y, c=colors[mean_res[i][0]], label=mean_res[i][0])
index_maxf1 = list(map(lambda x: x[2], mean_res)).index(max(map(lambda x: x[2], mean_res)))
ax.annotate(str(round(mean_res[index_maxf1][2], 3)), (mean_res[index_maxf1][3], mean_res[index_maxf1][4]))
ax.axis((0,1,0,1))
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Test scores: Recall vs. Precision")
plt.savefig("test_scores_recall_vs_precision_{}.png".format(dataname), transparent=True)
plt.show()

#plot mean train scores
fig, ax = plt.subplots(figsize=(15,15))
ax.plot([], [], ' ', label="Top algorithm is annotated with F1 score")
for i in range(len(mean_res)):
    x = mean_res[i][6]
    y = mean_res[i][7]
    ax.scatter(x, y, c=colors[mean_res[i][0]], label=mean_res[i][0])
index_maxf1 = list(map(lambda x: x[5], mean_res)).index(max(map(lambda x: x[5], mean_res)))
ax.annotate(str(round(mean_res[index_maxf1][5], 3)), (mean_res[index_maxf1][6], mean_res[index_maxf1][7]))
ax.axis((0,1,0,1))
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Train scores: Recall vs. Precision")
plt.savefig("train_scores_recall_vs_precision_{}.png".format(dataname), transparent=True)
plt.show()
