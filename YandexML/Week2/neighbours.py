import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data', header=None)

# Splitting classes and parameters
classes = data[0]
params = data.drop(columns=0)

# Tasks 1 and 2
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for k in range(1,50):
    clf = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(clf, X=params.values, y=classes.values, cv=kf, scoring='accuracy')
    cv_scores.append(score.mean())

optimal_k = range(1,50)[cv_scores.index(max(cv_scores))]

# Print answers to file
task = 1
f = open(f'neighbours_{task:d}.txt', 'w')
f.write(str(optimal_k))
f.close()
task = 2
f = open(f'neighbours_{task:d}.txt', 'w')
f.write(str(round(max(cv_scores), 2)))
f.close()

#End of tasks 1 and 2

# Tasks 3 and 4

scaled_params = scale(X=params)

sc_cv_scores = []

for k in range(1,50):
    clf = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(clf, X=scaled_params, y=classes.values, cv=kf, scoring='accuracy')
    sc_cv_scores.append(score.mean())

sc_optimal_k = range(1,50)[sc_cv_scores.index(max(sc_cv_scores))]

# Print answers to file
task = 3
f = open(f'neighbours_{task:d}.txt', 'w')
f.write(str(sc_optimal_k))
f.close()
task = 4
f = open(f'neighbours_{task:d}.txt', 'w')
f.write(str(round(max(sc_cv_scores), 2)))
f.close()

#End of tasks 3 and 4




