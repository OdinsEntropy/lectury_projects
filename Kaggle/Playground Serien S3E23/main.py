import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


#%% Get Input data and split it up 
train_data = pd.read_csv("./input/train.csv", index_col='id')
test_data_x = pd.read_csv("./input/test.csv", index_col='id')


train_data_x = train_data.drop("defects", axis=1)
train_data_y = train_data['defects']


#%% Traindata split

X_train, X_val, y_train, y_val = train_test_split(train_data_x, train_data_y, test_size=0.33, random_state=42)

#%% Setup model

clf = RandomForestClassifier(max_depth=3, random_state=2)
clf.fit(X_train,y_train)
print("Accuracy: ", clf.score(X_val, y_val))
print("ROC: ", roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))



#%%

#fit on all data now
clf.fit(train_data_x,train_data_y)


print(clf.score(X_val, y_val))

output_pd = pd.DataFrame({'defects':clf.predict_proba(test_data_x)[:, 1]})
output_pd.to_csv('test_pred.csv',index=True)