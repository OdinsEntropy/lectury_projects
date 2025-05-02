# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:18:14 2023

@author: Eric
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier


smoking = pd.read_csv('input/smoking.csv', index_col='ID')
train = pd.read_csv('input/train.csv', index_col='id')
test = pd.read_csv('input/test.csv', index_col='id')

from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.parallel import Parallel, delayed
from sklearn.preprocessing import OrdinalEncoder

def mi(x,y,n_iter=5):
    X = np.array(x).reshape((-1,1))
    y = np.array(y)
    if X.dtype == 'object':
        X = OrdinalEncoder().fit_transform(X)
    scores = Parallel(n_jobs=4)(delayed(mutual_info_classif)(X,y,random_state=42+i) for i in range(n_iter))
    return np.array([np.mean(scores), np.std(scores)])

def theil_u(train,target,comment=''):
    print(F'*** U(y|x) in % {comment} ***')
    e = mi(train[target],train[target])[0]
    for c in train.columns:
        if c != target:
            mu, sigma = mi(train[c],train[target])/e*100
            print(F"{c}: {mu:.2f} ± {sigma:.2f}")

theil_u(smoking,target='smoking',comment='(Original data)')


#%%
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

kfold = RepeatedStratifiedKFold(n_splits=10,n_repeats=5)
X_0 = smoking.drop(['gender','oral','tartar','smoking'],axis=1)
y_0 = smoking['gender'].map({'F':0,'M':1})

results = cross_validate(XGBClassifier(n_jobs=4,random_state=0),
                         X_0,y_0,
                         scoring='roc_auc',
                         cv=kfold,n_jobs=1,
                         return_estimator=True)

results['test_score'].mean(), results['test_score'].std()

#%%

train['gender'] = 0
cols = list(test.columns)

for clf in results['estimator']:
    train['gender'] += clf.predict_proba(train[cols])[:,1]
train['gender'] /= len(results['estimator'])
#theil_u(train,target='smoking',comment='(train)')

#%%

test['gender'] = 0
#cols = list(test.columns)

for clf in results['estimator']:
    test['gender'] += clf.predict_proba(test[cols])[:,1]
test['gender'] /= len(results['estimator'])
#theil_u(train,target='smoking',comment='(train)')



#%%
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline


models = {       'LGBM1C': LGBMClassifier(**{'device'           : "cpu",
                           'objective'         : 'binary',
                           'metric'            : 'auc',
                           'boosting_type'     : 'gbdt',
                           'random_state'      : 23,
                           'colsample_bytree'  : 0.35,
                           'subsample'         : 0.45,
                           'learning_rate'     : 0.05,
                           'max_depth'         : 8,
                           'n_estimators'      : 1000,
                           'num_leaves'        : 175,                    
                           'reg_alpha'         : 0.01,
                           'reg_lambda'        : 1.75,
                           'verbose'           : -1,
                        }
                     )}


classifier_obj = make_pipeline(models["LGBM1C"])


auc = cross_validate(classifier_obj, train.drop('smoking', axis=1),train.smoking, cv=5, n_jobs=-1, scoring='roc_auc')
print(np.mean(auc["test_score"]))

#%%

models["LGBM1C"].fit(train.drop('smoking', axis=1),train.smoking)