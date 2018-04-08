import numpy as np
import pandas
import sklearn

# Reading data
data = pandas.read_csv('classification.csv')

true = data.values[:, :1]
pred = data.values[:, 1:]


# Task 1
tp = true[(true == 1) & (pred == 1)].size
fp = true[(true == 0) & (pred == 1)].size
fn = true[(true == 1) & (pred == 0)].size
tn = true[(true == 0) & (pred == 0)].size

# Print answers to file
task = 1
f = open(f'metrics_answer_{task:d}.txt', 'w')
f.write(' '.join(str(x) for x in [tp, fp, fn, tn]))
f.close()


# Task 2
#Finding metrics
accuracy = round(sklearn.metrics.accuracy_score(true, pred), 2)
precision = round(sklearn.metrics.precision_score(true, pred), 2)
recall = round(sklearn.metrics.recall_score(true, pred), 2)
f_meas = round(sklearn.metrics.f1_score(true, pred), 2)

# Print answers to file
task = 2
f = open(f'metrics_answer_{task:d}.txt', 'w')
f.write(' '.join(str(x) for x in [accuracy, precision, recall, f_meas]))
f.close()


#Task 3
# Reading data about classifiers
class_data = pandas.read_csv('scores.csv')

#Reading true result and classifiers predictions
true = class_data['true']
logreg = class_data['score_logreg']
svm = class_data['score_svm']
knn = class_data['score_knn']
tree = class_data['score_tree']

#Finding AUC-ROC on each classifier
logreg_score = sklearn.metrics.roc_auc_score(true, logreg)
svm_score = sklearn.metrics.roc_auc_score(true, svm)
knn_score = sklearn.metrics.roc_auc_score(true, knn)
tree_score = sklearn.metrics.roc_auc_score(true, tree)

#Fingding winner column name
col_name = class_data.columns[np.argmax([logreg_score, svm_score, knn_score, tree_score]) + 1]

# Print answers to file
task = 3
f = open(f'metrics_answer_{task:d}.txt', 'w')
f.write(col_name)
f.close()


#Task 4
#Finding max on each of curves

#LogReg
curve = sklearn.metrics.precision_recall_curve(true, logreg)
precision, recall = curve[0], curve[1]
logreg_max = np.max(precision[recall >= 0.7])

#SVM
curve = sklearn.metrics.precision_recall_curve(true, svm)
precision, recall = curve[0], curve[1]
svm_max = np.max(precision[recall >= 0.7])

#K-heighbours
curve = sklearn.metrics.precision_recall_curve(true, knn)
precision, recall = curve[0], curve[1]
knn_max = np.max(precision[recall >= 0.7])

#Tree
curve = sklearn.metrics.precision_recall_curve(true, tree)
precision, recall = curve[0], curve[1]
tree_max = np.max(precision[recall >= 0.7])

#Fingding winner column name
col_name = class_data.columns[np.argmax([logreg_max, svm_max, knn_max, tree_max]) + 1]

# Print answers to file
task = 4
f = open(f'metrics_answer_{task:d}.txt', 'w')
f.write(col_name)
f.close()