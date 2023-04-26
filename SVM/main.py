import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate a 3D dataset
X, y = make_classification(n_features=3, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=100, random_state=42)

# Create an SVM classifier with RBF kernel
svm = SVC(kernel='rbf', C=1, gamma=0.1)

# Train the classifier on the data
svm.fit(X, y)

# Plot the decision boundary in a 3D feature space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid of points in the 3D feature space
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))

# Evaluate the SVM classifier on the meshgrid points
Z = svm.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]).reshape(xx.shape)

# Plot the decision boundary as a surface in the 3D feature space
ax.plot_surface(xx, yy, zz, alpha=0.2, facecolors=plt.cm.coolwarm(Z))
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.show()
