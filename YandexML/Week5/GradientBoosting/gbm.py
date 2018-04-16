import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from matplotlib.pyplot import plot

# Reading data
data = pandas.read_csv('gbm-data.csv')

# Get targets and params
targets = data['Activity'].values
params = data.drop(columns='Activity').values

# Selecting train and test parts
X_train, X_test, y_train, y_test = train_test_split(params, targets, test_size = 0.8, random_state = 241)


# Classifier    
for rate in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(learning_rate = rate, n_estimators = 250, verbose = True, random_state = 241)
    clf.fit(X_train, y_train)
    
    train_loss = []
    for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
        sigmoid = 1 / (1 + np.exp(-y_pred))
        train_loss.append(log_loss(y_train, sigmoid))
    test_loss = []
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        sigmoid = 1 / (1 + np.exp(-y_pred))
        test_loss.append(log_loss(y_test, sigmoid))
    plot(train_loss, color='green')
    plot(test_loss, color='red')
    
    if (rate == 0.2):
        min = np.min(test_loss)
        iter = np.argmin(test_loss) + 1
    
# Answer for task 2
task = 2
f = open(f'gbm_answer_{task:d}.txt', 'w')
f.write(' '.join(str(x) for x in [round(min, 2), iter]))   
f.close()

# Learn at best iter count
clf = GradientBoostingClassifier(learning_rate = 0.2, n_estimators = iter, verbose = True, random_state = 241)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
loss = log_loss(y_test, pred)

# Answer for task 3
task = 3
f = open(f'gbm_answer_{task:d}.txt', 'w')
f.write(str(round(loss, 2)))   
f.close()