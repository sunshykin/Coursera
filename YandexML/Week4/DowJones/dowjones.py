import numpy as np
import pandas
from sklearn.decomposition import PCA

# Reading data
data = pandas.read_csv('close_prices.csv')
dowjones_data = pandas.read_csv('djia_index.csv')
X = data.drop(columns='date')

# Finding out components count for explaining 90% of dispersion
num = 0

for n in range(31):
    pca = PCA(n_components = n)
    pca.fit(X)
    percent = np.sum(pca.explained_variance_ratio_)
    if (percent >= 0.9):
        num = n
        break

# Answer for task 1
task = 1
f = open(f'dowjones_answer_{task:d}.txt', 'w')
f.write(str(num))
f.close()


# Setting up pca to 10 components
pca = PCA(n_components = 10)
pca.fit(X)
X_transformed = pca.transform(X)
first_comp = X_transformed[:, 0]

# Finding correlation with DowJones Index
corr_coef = np.corrcoef(first_comp, dowjones_data['^DJI'])[0,1]

# Answer for task 2
task = 2
f = open(f'dowjones_answer_{task:d}.txt', 'w')
f.write(str(round(corr_coef, 2)))   
f.close()


# Finding companies' weights
weights = pca.components_
# Adding 1 cause of date column
company = data.columns[np.argmax(weights[0]) + 1]

# Answer for task 3
task = 3
f = open(f'dowjones_answer_{task:d}.txt', 'w')
f.write(company)   
f.close()