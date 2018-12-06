import numpy as numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("dataset/train.csv").as_matrix()
print(data)
classifier = DecisionTreeClassifier()

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
