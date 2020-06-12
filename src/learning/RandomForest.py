#!/usr/bin/python3

import sqlite3 as sq
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from operator import itemgetter
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns

data=np.genfromtxt('./data/adult.data', delimiter=', ', unpack=True, dtype="S")
print(data[:,0])
le = preprocessing.LabelEncoder()


data_frame=pd.DataFrame({
    'age':data[0,:].astype(int),
    'workclass':le.fit_transform(data[1,:]),
    'fnlwgt':data[2,:].astype(int),
    'education':le.fit_transform(data[3,:]),
    'education-num':data[4,:].astype(int),
    'marital-status':le.fit_transform(data[5,:]),
    'occupation':le.fit_transform(data[6,:]),
    'relationship':le.fit_transform(data[7,:]),
    'race':le.fit_transform(data[8,:]),
    'sex':le.fit_transform(data[9,:]),
    'capital-gain':data[10,:].astype(float),
    'capital-loss':data[11,:].astype(float),
    'hours-per-week':data[12,:].astype(float),
    'native-country':le.fit_transform(data[13,:]),
    'income': le.fit_transform(data[14,:])
})
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
x=data_frame[feature_names]
y=data_frame['income']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
print(feature_imp)

sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()