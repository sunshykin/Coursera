import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV 

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

texts = newsgroups.data
classes = newsgroups.target

vect = TfidfVectorizer()
train = vect.fit_transform(texts)

feature_mappings = vect.get_feature_names()

# Checking out the most suitable C
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(train, classes)

max = 0
optimal_C = 0

for a in gs.grid_scores_:
    if a.mean_validation_score > max:
        max = a.mean_validation_score
        optimal_C = a.parameters['C']

# Reinit classifier with optimal C
clf = SVC(C=optimal_C, kernel='linear', random_state=241)
clf.fit(train, classes)

# Finding top10 used words
word_indexes = np.argsort(np.abs(clf.coef_.toarray()[0]))[-10:]
words = [str(feature_mappings[i]) for i in word_indexes]


# Answer
f = open('svm_texts_answer.txt', 'w')
f.write(' '.join(w for w in sorted(words)))
f.close()