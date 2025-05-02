# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:40:45 2023

@author: Eric
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier,HistGradientBoostingClassifier,BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Discretization, BatchNormalization, Normalization

#%% Get Input data and split it up 

train = pd.read_csv('./input/train.csv', index_col='id')
test = pd.read_csv('./input/test.csv', index_col='id')
original = pd.read_csv('./input/jm1.csv',na_values=['?'])

#%% Optimized

#0.7924158324959047 soft weights
#0.7925703907000827
#0.7927093466284703


ensemble = VotingClassifier(
    [
     ('RF', make_pipeline(FunctionTransformer(np.log1p),
                          RandomForestClassifier( max_depth=9, n_estimators=431, criterion='entropy', min_samples_leaf=28, max_features=1.0, random_state=23, n_jobs=-1))),

     ('et', make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])], remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(n_estimators=189,min_samples_leaf=34, criterion='gini', max_depth=12,  max_features=1.0,random_state=23,n_jobs=-1))),
     
     ('hgb', HistGradientBoostingClassifier(random_state=23)),
     
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=745, random_state=23),
                          StandardScaler(),
                          LogisticRegression(dual=False, C=0.0009095968332141859, max_iter=1500, class_weight='balanced', random_state=23,n_jobs=-1))),
     
     ('GBC', GradientBoostingClassifier(n_estimators=24, learning_rate=0.19988299934404352, max_depth=4, random_state=23)),
     
     ('LGBM', make_pipeline(FunctionTransformer(np.log1p),
                            lgb.LGBMClassifier(boosting_type='dart', n_estimators=725, class_weight='balanced', num_leaves=5, max_depth=3, n_jobs=-1, random_state=23))),
      
     ('XGB', XGBClassifier(n_estimators=243, max_depth=2, learning_rate=0.16643025468333306, n_jobs=-1, random_state=23)),
     
     ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,task_type="GPU",random_seed=23))
     ], voting='soft')

#%%
ensemble = StackingClassifier(
    [
     ('RF', make_pipeline(FunctionTransformer(np.log1p),
                          RandomForestClassifier( max_depth=9, n_estimators=431, criterion='entropy', min_samples_leaf=28, max_features=1.0, random_state=23, n_jobs=-1))),

     ('et', make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])], remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(n_estimators=189,min_samples_leaf=34, criterion='gini', max_depth=12,  max_features=1.0,random_state=23,n_jobs=-1))),
     
     ('hgb', HistGradientBoostingClassifier(random_state=23)),
     
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=745, random_state=23),
                          StandardScaler(),
                          LogisticRegression(dual=False, C=0.0009095968332141859, max_iter=1500, class_weight='balanced', random_state=23,n_jobs=-1))),
     
     ('GBC', GradientBoostingClassifier(n_estimators=24, learning_rate=0.19988299934404352, max_depth=4, random_state=23)),
     
     ('LGBM', make_pipeline(FunctionTransformer(np.log1p),
                            lgb.LGBMClassifier(boosting_type='dart', n_estimators=725, class_weight='balanced', num_leaves=5, max_depth=3, n_jobs=-1, random_state=23))),
      
     ('XGB', XGBClassifier(n_estimators=243, max_depth=2, learning_rate=0.16643025468333306, n_jobs=-1, random_state=23)),
     
     ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,random_seed=23))],
    n_jobs=-1, final_estimator=LogisticRegression(), stack_method='predict_proba')

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('defects', axis=1),train.defects, stratify=train.defects, random_state=23)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)



#%%

clf = [
     ('RF', make_pipeline(FunctionTransformer(np.log1p),
                          RandomForestClassifier( max_depth=9, n_estimators=431, criterion='entropy', min_samples_leaf=28, max_features=1.0, random_state=23, n_jobs=-1))),

     ('et', make_pipeline(ColumnTransformer([('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])], remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(n_estimators=189,min_samples_leaf=34, criterion='gini', max_depth=12,  max_features=1.0,random_state=23,n_jobs=-1))),
     
     ('hgb', HistGradientBoostingClassifier(random_state=23)),
     
     ('ny', make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=745, random_state=23),
                          StandardScaler(),
                          LogisticRegression(dual=False, C=0.0009095968332141859, max_iter=1500, class_weight='balanced', random_state=23,n_jobs=-1))),
     
     ('GBC', GradientBoostingClassifier(n_estimators=24, learning_rate=0.19988299934404352, max_depth=4, random_state=23)),
     
     ('LGBM', make_pipeline(FunctionTransformer(np.log1p),
                            lgb.LGBMClassifier(boosting_type='dart', n_estimators=725, class_weight='balanced', num_leaves=5, max_depth=3, n_jobs=-1, random_state=23))),
      
     ('XGB', XGBClassifier(n_estimators=243, max_depth=2, learning_rate=0.16643025468333306, n_jobs=-1, random_state=23)),
     
     ('CATB', CatBoostClassifier(n_estimators=561, max_depth=4, learning_rate=0.05823290251374662,random_seed=23,task_type="GPU"))]
    
trained= []
for i in clf:
#    trained.append(i[1].fit(train.drop('defects', axis=1),train.defects))
    trained.append(i[1].fit(X_train,y_train))
    
#%% Data augmentation

new_data = pd.DataFrame()
for i, j in zip(clf,trained):
    new_data[i[0]] = j.predict_proba(X_train)[:,1]
    
new_data_test = pd.DataFrame()
for i, j in zip(clf,trained):
    new_data_test[i[0]] = j.predict_proba(X_test)[:,1]


#new_data = pd.concat([new_data,new_data.mean(axis=1)], ignore_index=False, axis=1)
#new_data_test = pd.concat([new_data_test,new_data_test.mean(axis=1)], ignore_index=True, axis=1)

#std_trans = StandardScaler().set_output(transform="pandas")

#new_data = pd.concat([new_data,std_trans.fit_transform(FunctionTransformer(np.log1p).fit_transform(X_train))], ignore_index=False, axis=1)
#new_data_test = pd.concat([new_data_test,std_trans.fit_transform(FunctionTransformer(np.log1p).fit_transform(X_test))], ignore_index=True, axis=1)


#%% CNN Model
model = Sequential()
model.add(Dense(4,  activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%% CNN Model Best one with only percentages
model = Sequential()
model.add(Dense(4, activation='softplus'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
#%% CNN Model
model = Sequential()
model.add(Dense(8, activation='softplus',kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',kernel_regularizer='l2'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%% CNN Model
model = Sequential()
model.add(Dense(2, activation='sigmoid', use_bias=False))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

#%%    
#history = model.fit(new_data,y_train, epochs=500, batch_size=500, validation_split=0.3)
history = model.fit(new_data,y_train, epochs=30, batch_size=500, validation_data=(new_data_test,y_test))

#%%



auc = cross_validate(ensemble, train.drop('defects', axis=1),train.defects, cv=5, scoring=['roc_auc'])
result = np.mean(auc["test_roc_auc"])
#%% Train ensamble without weights
abc = ensemble.fit(train.drop('defects', axis=1),train.defects).predict_proba(test)

#%% Transform Data
transf = ensemble.predict_proba(train.drop('defects', axis=1))[:, 1]



auc = cross_validate(ensemble, train.drop('defects', axis=1),train.defects, cv=5, scoring=['roc_auc'])
result = np.mean(auc["test_roc_auc"])

#%%
output_pd = pd.DataFrame({'defects':ensemble.predict_proba(test)[:, 1]}, index=test.index)
output_pd.to_csv(f"test_stackingclassifier.csv",index=True)


#%%
classifier_obj = LogisticRegression(random_state=23,n_jobs=-1, class_weight='balanced')

auc = cross_validate(classifier_obj, new_data,y_train, cv=5, scoring='roc_auc')
print(np.mean(auc["test_score"]))


#%% 0.7918493680476315
from sklearn.linear_model import SGDClassifier

classifier_obj = LogisticRegression(random_state=23,n_jobs=-1, class_weight='balanced', C=0.001, verbose=1)
#classifier_obj = SGDClassifier(n_jobs=-1, loss="log_loss", penalty='l2')


abc = classifier_obj.fit(new_data,y_train).predict_proba(new_data_test)
print(roc_auc_score(y_test, abc[:,1]))


#%% 0.7917663329838279

from scikeras.wrappers import KerasClassifier

def create_model():

    model = Sequential()
    model.add(Dense(4, input_dim=8,activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    return model

classifier_obj = KerasClassifier(model=create_model, epochs=12, batch_size=500, verbose=0)

abc = classifier_obj.fit(new_data,y_train).predict_proba(new_data_test)
print(roc_auc_score(y_test, abc[:,1]))


#%% 0.7918466002121715

print(roc_auc_score(y_test, np.mean(new_data_test*1/5, axis=1)))

#%% 0.7918610740523799
print(roc_auc_score(y_test, np.mean(new_data_test*[0,5,1,1,1,5,5,1], axis=1)))

#%% 0.7918866809377674
print(roc_auc_score(y_test, np.mean(new_data_test*[3,4,0,3,2,5,3,5], axis=1)))


#%%0.7919190046722648
print(roc_auc_score(y_test, np.mean(new_data_test*[3,5,0,2,0,4,4,5], axis=1)))

 

