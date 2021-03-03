#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
from sklearn import metrics
import pandas as pd
import torch.utils.data as Data
import sklearn
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


df = pd.read_csv("./beijing_cate2id.csv")

shape1 = df.shape[0]
baseline_train_df = df.iloc[0:int(0.6 * shape1)]
baseline_val_df = df.iloc[int(0.6 * shape1):int(0.8 * shape1)]
baseline_test_df = df.iloc[int(0.8 * shape1):]


qt = QuantileTransformer(output_distribution="normal").fit(df.loc[:, df.columns != 'scene'])

x_train = baseline_train_df.loc[:, baseline_train_df.columns != 'scene']
a  = x_train.columns
x_train = qt.transform(x_train)
x_train = pd.DataFrame(x_train)
x_train.columns = a
y_train = pd.Categorical(baseline_train_df.scene).codes

x_test = baseline_test_df.loc[:, baseline_test_df.columns != 'scene']
a  = x_test.columns
x_test = qt.transform(x_test)
x_test = pd.DataFrame(x_test)
x_test.columns = a
y_test = pd.Categorical(baseline_test_df.scene).codes

x_val = baseline_val_df.loc[:, baseline_val_df.columns != 'scene']
a  = x_val.columns
x_val = qt.transform(x_val)
x_val = pd.DataFrame(x_val)
x_val.columns = a
y_val = pd.Categorical(baseline_val_df.scene).codes


x_train = np.array(x_train)
y_train = np.array(y_train).reshape(-1, 1) # (-1,1)
x_test = np.array(x_test)
y_test = np.array(y_test).reshape(-1, 1)
x_val = np.array(x_val)
y_val = np.array(y_val).reshape(-1, 1)


def metric(pred, true, proba):
    
    # acc
    sz = pred.shape[0]
    flag = 0
    for i in range(sz):
        if pred[i] == true[i]:
            flag += 1
    acc = flag/sz
    # precision recall
    p = precision_score(true, pred, average='macro',zero_division=1)
    r = recall_score(true, pred, average='macro')
    # roc and auc

    try:
        auc = metrics.roc_auc_score(true, proba, multi_class="ovr", average='macro', labels=[0,1,2,3])
    except:
        auc = None
    return acc, p, r, auc


# 1. DT
model = tree.DecisionTreeClassifier()
# 2. SVM
# model = SVC(kernel="linear", C=0.1, probability=True, tol=0.01)


model = model.fit(x_train, y_train)
print("train:")
p_ytrain = model.predict(x_train)
prob = model.predict_proba(x_train)
acc, p, r, auc = metric(p_ytrain, y_train, prob)
print(acc, p, r, auc)

print("val:")
p_yval = model.predict(x_val)
prob = model.predict_proba(x_val)
acc, p, r, auc = metric(p_yval, y_val, prob)
print(acc, p, r, auc)

print("test:")
p_ytest = model.predict(x_test)
prob = model.predict_proba(x_test)
acc, p, r, auc = metric(p_ytest, y_test, prob)
print(acc, p, r, auc)

