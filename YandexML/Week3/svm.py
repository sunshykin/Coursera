import numpy as np
import pandas
from sklearn.svm import SVC

# Reading data
data = pandas.read_csv('svm-data.csv', header=None)

# Get targets and params
targets = data[0]
params = data.drop(columns=0)

# Creating classifier and fitting data
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(params, targets)

# Answer
f = open('svm_answer.txt', 'w')
# Adding 1 to s cause we need to get numbers from [1..10] not [0..9]
f.write(' '.join(str(s+1) for s in clf.support_))
f.close()