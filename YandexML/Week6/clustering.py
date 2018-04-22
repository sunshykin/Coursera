import numpy as np
from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import pylab

# Loading image
image = imread('parrots.jpg')

# Converting from 0..255 to 0..1
float_image = np.array(img_as_float(image))

x, y, z = float_image.shape
objects_features_matrix = np.reshape(float_image, (x * y, z))

# Making clusterizer
clst = KMeans(init='k-means++', random_state=241).fit(objects_features_matrix)
labels = clst.labels_

color_median = np.zeros((6, 3))
for clust in range(6):
    color_median[clust] = np.median(objects_features_matrix[np.where(labels == clust)], axis=0)

#img_pred = [clst.cluster_centers_[i] for i in clst.fit_predict(objects_features_matrix)]
print(clst.cluster_centers_)

