import pandas
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Starting timer
program_start = dt.datetime.now()

features = pandas.read_csv('./data/features.csv', index_col='match_id')
 
#--------------------- Data Preparation ---------------------
# Getting targets and params for train data
drop_cols = features.columns.tolist()[-6:]
drop_cols.append('start_time')

params = features.drop(columns=drop_cols)
targets = features['radiant_win']


# Get count of not NaN values
counts = params.count()
# Params which are not fully filled at 5min
not_full_train = counts[counts < len(features)].index.tolist()

# Filled NaN with mean of columns
params_wo_nan = params.fillna(params.mean())
#--------------------------- End -------------------------


#--------------------- Gradient Boosting ---------------------
time = []
accur = []
for n in [10, 20, 30, 40, 50]:
    start = dt.datetime.now()
    
    gb_clf = GradientBoostingClassifier(n_estimators = n, random_state = 21)
    valid = KFold(n_splits=5, random_state = 21, shuffle = True)
    
    auc_roc = []
    built_in = []
    for train, test in valid.split(params_wo_nan):
        X_train, X_test = params_wo_nan.iloc[train], params_wo_nan.iloc[test]
        y_train, y_test = targets.iloc[train], targets.iloc[test]
        gb_clf.fit(X_train, y_train)
        
        # Built-In score
        built_in.append(gb_clf.score(X_test, y_test))
        
        #AUC-ROC score
        y_pred = gb_clf.predict_proba(X_test)[:, 1]
        auc_roc.append(roc_auc_score(y_test, y_pred))
    print('N=%d | Time=%s | Built-in score=%f | AUC-ROC score=%f' % (n, dt.datetime.now() - start, np.mean(built_in), np.mean(auc_roc)))
#--------------------------- End -------------------------
    

def LogRegression(arr, params, targets):
    for c in arr:
        start = dt.datetime.now()
        
        lr_clf = LogisticRegression(penalty='l2', tol=0.0001, C=c, solver='sag', max_iter=100)
        valid = KFold(n_splits=5, random_state = 21, shuffle = True)
        
        auc_roc = []
        built_in = []
        for train, test in valid.split(params):
            X_train, X_test = params[train], params[test]
            y_train, y_test = targets.iloc[train], targets.iloc[test]
            lr_clf.fit(X_train, y_train)
            
            # Built-In score
            built_in.append(lr_clf.score(X_test, y_test))
            
            #AUC-ROC score
            y_pred = lr_clf.predict_proba(X_test)[:, 1]
            auc_roc.append(roc_auc_score(y_test, y_pred))
        print('C=%f | Time=%s | Built-in score=%f | AUC-ROC score=%f' % (c, dt.datetime.now() - start, np.mean(built_in), np.mean(auc_roc)))

def GetWordBag(data, n):
    pick = np.zeros((data.shape[0], n))
    
    for i, match_id in enumerate(data.index):
        for p in range(5):
            pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return pick

 
#--------------------- Logistic Regression ---------------------    
# Array of C values
c_array = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.0, 0.1), np.arange(1, 10, 1), np.arange(10, 101, 10)])

sc = StandardScaler()
params_scaled = sc.fit_transform(params_wo_nan)
LogRegression(c_array, params_scaled, targets)


# Droping heroes and lobby_type columns
drop_cols = [col for col in params_wo_nan.columns if 'hero' in col]
drop_cols.append('lobby_type')

sc = StandardScaler()
params_scaled = sc.fit_transform(params_wo_nan.drop(columns=drop_cols))
LogRegression(c_array, params_scaled, targets)


# Word Bag
hero_count = np.max(np.unique(params_wo_nan[drop_cols[0]])) 
X_pick = GetWordBag(params_wo_nan, hero_count)

sc = StandardScaler()
params_scaled = sc.fit_transform(params_wo_nan.drop(columns=drop_cols))
params_w_bag = np.hstack([params_scaled, X_pick])

LogRegression(c_array, params_w_bag, targets)
#--------------------------- End -------------------------


#--------------------- Predicting on test data ---------------------
# Preparing data
test_features = pandas.read_csv('./data/features_test.csv', index_col='match_id')
test_params = test_features.drop(columns='start_time')
test_params = test_params.fillna(test_params.mean())

test_bag = GetWordBag(X_test, hero_count)

test_params_scaled = StandardScaler().fit_transform(test_params.drop(columns=drop_cols))
test_params_w_bag = np.hstack([test_params_scaled, test_bag])

# Fitting LinReg with best C
best_clf = LogisticRegression(penalty='l2', tol=0.0001, C=0.1, solver='sag', max_iter=100)
best_clf.fit(params_w_bag, targets)

predictions = best_clf.predict_proba(test_params_w_bag)

probs = []
for prob_0, prob_1 in predictions:
    if prob_0 > prob_1:
        probs.append(prob_0)
    else:
        probs.append(prob_1)
        
print('Min probability=%f | Max probability=%f' % (np.min(probs), np.max(probs)))
#--------------------------- End -------------------------


print('End of file. Time elapsed:', dt.datetime.now() - program_start)