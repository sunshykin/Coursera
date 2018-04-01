import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train_data = pandas.read_csv('perceptron-train.csv', header=None)
test_data = pandas.read_csv('perceptron-test.csv', header=None)

# Get targets and params for train data
train_targets = train_data[0]
train_params = train_data.drop(columns=0)

# Get targets and params for test data
test_targets = test_data[0]
test_params = test_data.drop(columns=0)

# Training perceptron without normalizing
clf = Perceptron(random_state=241)
clf.fit(X=train_params, y=train_targets)
preditcts = clf.predict(test_params)
accuracy = accuracy_score(test_targets, preditcts)

# Normalizing data
scaler = StandardScaler()
sc_train_params = scaler.fit_transform(train_params)
sc_test_params = scaler.transform(test_params)

# Training with normalizing
sc_clf = Perceptron(random_state=241)
sc_clf.fit(X=sc_train_params, y=train_targets)
sc_preditcts = sc_clf.predict(sc_test_params)
sc_accuracy = accuracy_score(test_targets, sc_preditcts)

# Finding difference between 2 accuracies
differ = round(sc_accuracy - accuracy, 3)

# Answer
f = open('perceptron_answer.txt', 'w')
f.write(str(differ))
f.close()