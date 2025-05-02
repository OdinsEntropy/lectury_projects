import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


#%% Get Input data and split it up 
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
test_data_x = test_data


train_data_x = train_data.drop("Survived", axis=1)
train_data_y = train_data['Survived']

#%% Prepare Data 

#replace nan for age and fare

train_data_x["Age"].fillna(train_data_x["Age"].mean(), inplace=True)
test_data_x["Age"].fillna(train_data_x["Age"].mean(), inplace=True)


train_data_x["Fare"].fillna(train_data_x["Fare"].mean(), inplace=True)
test_data_x["Fare"].fillna(train_data_x["Fare"].mean(), inplace=True)

#encode variable embarked
train_data_x = pd.get_dummies(train_data_x, columns = ['Sex','Embarked'])
test_data_x = pd.get_dummies(test_data_x, columns = ['Sex','Embarked'])


#Add Features
train_data_x['Family_Size']=train_data_x['SibSp']+train_data_x['Parch']
test_data_x['Family_Size']=test_data_x['SibSp']+test_data_x['Parch']


train_data_x['Fare_Per_Person']=train_data_x['Fare']/(train_data_x['Family_Size']+1)
test_data_x['Fare_Per_Person']=test_data_x['Fare']/(test_data_x['Family_Size']+1)


title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan

train_data_x['Title']=train_data_x['Name'].map(lambda x: substrings_in_string(x, title_list))
test_data_x['Title']=test_data_x['Name'].map(lambda x: substrings_in_string(x, title_list))

 
#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex_male']==True:
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
    
train_data_x['Title']=train_data_x.apply(replace_titles, axis=1)
train_data_x = pd.get_dummies(train_data_x, columns = ['Title'])

test_data_x['Title']=test_data_x.apply(replace_titles, axis=1)
test_data_x = pd.get_dummies(test_data_x, columns = ['Title'])




#Drop not used data
train_data_x.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test_data_x.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

#%% Traindata split

X_train, X_val, y_train, y_val = train_test_split(train_data_x, train_data_y, test_size=0.33, random_state=42)

#%% Setup model

clf = RandomForestClassifier(max_depth=3, random_state=2)
clf.fit(X_train,y_train)
print(clf.score(X_val, y_val))


#%%

#fit on all data now
clf.fit(train_data_x,train_data_y)


print(clf.score(X_val, y_val))

output_pd = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':clf.predict(test_data_x)}, columns=['PassengerId', 'Survived'])
output_pd.to_csv('output/output.csv',index=False)