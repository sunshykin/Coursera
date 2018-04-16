import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Reading data
data = pandas.read_csv('abalone.csv')

# Changing literal Sex to numbers
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Get targets and params
targets = data['Rings']
params = data.drop(columns='Rings')

# Count of trees where r2_score > 0.52
count = 0
# Learn forest with 1 to 50 trees
for i in np.arange(1, 50):
    rf = RandomForestRegressor(n_estimators = i, random_state = 1)
    valid = KFold(n_splits=5, random_state = 1, shuffle = True)
    score = []
    for train, test in valid.split(params):
        X_train, X_test = params.loc[train], params.loc[test]
        y_train, y_test = targets.loc[train], targets.loc[test]
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        score.append(r2_score(y_test, predictions))
    if np.average(score) > 0.52:
        count = i
        break

# Answer
f = open('forest_answer.txt', 'w')
f.write(str(count))
f.close()