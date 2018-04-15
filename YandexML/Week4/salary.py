import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

# Reading train data
data_train = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')

# Getting targets
y_train = np.array(data_train['SalaryNormalized']).astype(np.float)


# Preparing data
# Getting to lower
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].str.lower()

# Replacing non-letter and non-digit symbols to spaces
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Using TF-IDF to get params
vect = TfidfVectorizer(min_df=5)
X_train = vect.fit_transform(data_train['FullDescription'])
X_test = vect.transform(data_test['FullDescription'])

# Replacing empty fields with NaN
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

# One-Hot encoding
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Combining
comb_train = hstack([X_train, X_train_categ])
comb_test = hstack([X_test, X_test_categ])

# Training Ridge classifier
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(comb_train, y_train)

answers = clf.predict(comb_test)

# Answer
f = open('salary_answer.txt', 'w')
f.write(' '.join(str(round(a, 2)) for a in answers))
f.close()