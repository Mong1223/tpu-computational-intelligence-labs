from sklearn import svm

import pandas as pd
import numpy as np
import random
df = pd.read_csv('heart.csv')
print(df.describe())
print(df.shape)
# df = df.drop(0)
# df[8] = df[8].astype(np.int64)
#
# from sklearn.preprocessing import StandardScaler
# data = StandardScaler().fit_transform(df)
#
# # divide the data into a training set and a test set
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(data, df[8], test_size=0.3, random_state=42)
#
#
# clf = svm.SVC()
# clf.fit(x_train, y_train)
#
# # a = clf.predict(x_test)
#
# print(clf.score(x_test, y_test))
# random_index = random.randrange(23)
# print(df.values[:,2])
#
# a = pd.DataFrame()
# b= [1,2,3]
# df.loc[len(df)] = b
# a.append(b)
# print(a)