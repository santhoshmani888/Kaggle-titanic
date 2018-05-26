# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 01:27:39 2018

@author: santh_000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data=pd.read_csv('train.csv')
data.head()
data.describe()
data['Age']=data['Age'].fillna(data['Age'].median())
dummy={'male':0,'female':1}
data['Sex']=data['Sex'].apply(lambda x:dummy[x])
data['Fare']=data['Fare'].fillna(data['Fare'].median())
data['Embarked']=data['Embarked'].fillna("S")
#nan_rows = data[data['Embarked'].isnull()]
data.loc[data["Embarked"]=="S","Embarked"] = 0
data.loc[data["Embarked"]=="C","Embarked"] = 1
data.loc[data["Embarked"]=="Q","Embarked"] = 2
X=data[['Age','Pclass','Sex','Fare','Embarked']].values
Y=data[['Survived']].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
from sklearn import linear_model
clf=linear_model.LogisticRegression()
result=clf.fit(X_train,Y_train)
#print (result.score(X_train,Y_train))
test=pd.read_csv('test.csv')
test['Age']=test['Age'].fillna(test['Age'].median())
test['Sex']=test['Sex'].apply(lambda x:dummy[x])
test['Fare']=test['Fare'].fillna(test['Fare'].median())
test['Embarked']=test['Embarked'].fillna("S")
#nan_rows = data[data['Embarked'].isnull()]
test.loc[test["Embarked"]=="S","Embarked"] = 0
test.loc[test["Embarked"]=="C","Embarked"] = 1
test.loc[test["Embarked"]=="Q","Embarked"] = 2
#nan_rows = test[test['Fare'].isnull()]
Xtest=test[['Age','Pclass','Sex','Fare','Embarked']].values
test_predict=clf.predict(Xtest)
sub=pd.DataFrame({
        "PassengerId":test['PassengerId'],
        "Survived":test_predict})
sub.to_csv('titanic.csv',index=False)


