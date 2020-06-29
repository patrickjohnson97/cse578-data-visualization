#!/usr/bin/python3
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

data=np.genfromtxt('./data/adult.data', delimiter=', ', unpack=True, dtype="S")
le = preprocessing.LabelEncoder()
# Exclude everything other than age, fnlwgt, capital gain, and education num
cleanse = np.delete(data,[1,3,5,6,7,8,9,11,12,13,14], 0)
input = np.transpose(cleanse)
y=le.fit_transform(data[14,:])
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(input)
incorrect = sum(abs(np.subtract(pred_y,y)))
print(abs(incorrect-len(y))/(len(y)))