import pandas
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = load_boston()

# Find targets and parameters
targets = data.target
params = data.data

sc_params = scale(params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for i in np.linspace(1.0, 10.0, num=200):
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i, metric='minkowski')
    score = cross_val_score(clf, X=sc_params, y=targets, cv=kf, scoring='neg_mean_squared_error')
    cv_scores.append(score.mean())
    
optimal_p = np.linspace(1.0, 10.0, num=200)[cv_scores.index(max(cv_scores))]

f = open(f'boston_answer.txt', 'w')
f.write(str(optimal_p))
f.close()