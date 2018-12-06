import numpy as numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # 83.....
from sklearn.ensemble import RandomForestClassifier # 92.7...

data = pd.read_csv("dataset/train.csv").as_matrix()
print(data)

# Default n_estimators=10
# Result 92.7..
# Current result 95.9..
classifier = RandomForestClassifier(n_estimators=100)

#Training dataset
xtrain = data[0:21000, 1:]
train_label = data[0:21000, 0]

classifier.fit(xtrain, train_label)

# Testing data
xtest = data[21000:, 1:]
actual_label = data[21000:, 0]

prediction = classifier.predict(xtest)

count = 0
for i in range(0, 21000):
    count += 1 if prediction[i] == actual_label[i] else 0
print("Accuracy=", (count/21000)*100)
