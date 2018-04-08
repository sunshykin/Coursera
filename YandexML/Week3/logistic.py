import numpy as np
import pandas
from sklearn.metrics import roc_auc_score

# Reading data
data = pandas.read_csv('data-logistic.csv', header=None)

x = data.drop(columns=0)
y = data[0]

def sigmoid(x): 
    return 1.0 / (1 + np.exp(-x))

def distance(a, b):
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

def l_regr(x, y, k, w, C, eps, iter_count):
    w1, w2 = w
    for i in range (iter_count):
        w1_new = w1 + k * np.mean(y * x[1] * (1 - sigmoid(y*(w1 * x[1] + w2 * x[2])))) - k * C * w1
        w2_new = w2 + k * np.mean(y * x[2] * (1 - sigmoid(y*(w1 * x[1] + w2 * x[2])))) - k * C * w2
        if distance((w1_new, w2_new), (w1, w2)) < eps:
            break
        w1, w2 = w1_new, w2_new
        
    return sigmoid(w1 * x[1] + w2 * x[2])


p_0  = l_regr(x, y, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
p_1  = l_regr(x, y, 0.1, [0.0, 0.0], 10, 0.00001, 10000)

answer = ' '.join([str(round(roc_auc_score(y, p_0), 3)), str(round(roc_auc_score(y, p_1), 3))])

f = open('logistic_answer.txt', 'w')
f.write(answer)
f.close()