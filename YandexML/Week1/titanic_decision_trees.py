import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# Drop not needed columns and NaN values
woNan = data.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']).dropna(axis=0, how='any')

# Find targets and parameters
targets = woNan['Survived']
params = woNan.drop(columns=['Survived'])

# Converting male to 0 and female to 1
params['Sex'] = (params['Sex'] == 'male').astype(int)

# Learning
clf = DecisionTreeClassifier(random_state=241)
clf.fit(params, targets)

# Find importances
importances = clf.feature_importances_

# Sort importances by DESC
sortedArgs = np.argsort(importances)[::-1]

#getting first and second indexes
first = params.columns[sortedArgs[0]]
second = params.columns[sortedArgs[1]]

# Writing the answer to file
f = open(f'titanic_decision_trees_answer.txt', 'w')
f.write(','.join([first, second]))
f.close()




