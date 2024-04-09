import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Perform PCA with 3 principal components
n_components = 3
pca = PCA(n_components=n_components)
X_r = pca.fit_transform(X)

# Eigenvalues and eigenvectors of the first 3 principal components
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Visualize the data in reduced dimensions
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_r[iris.target == i, 0], X_r[iris.target == i, 1], color=color, alpha=0.8, lw=lw, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()

# Plot the graph PC versus reconstruction error
n_components_range = range(1, n_components + 1)
explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(n_components_range, explained_var_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()
