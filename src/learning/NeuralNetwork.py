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
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as model_selection

raw=np.genfromtxt('./data/adult.data', delimiter=', ', unpack=True, dtype="S")
le = preprocessing.LabelEncoder()
# Exclude everything other than age, fnlwgt, capital gain, and education num
cleanse = np.delete(raw,[1,3,5,6,7,8,9,11,12,13,14], 0)
data = np.transpose(cleanse).astype(np.float64)
y=le.fit_transform(raw[14,:]).astype(np.float64)
X_train, X_test, y_train, y_test = train_test_split(data, y, train_size=0.7,test_size=0.3, random_state=0)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_observed = clf.predict(X_test)
incorrect = sum(abs(np.subtract(y_observed,y_test)))
print(abs(incorrect-len(y))/(len(y)))


