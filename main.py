import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import io, img_as_float

# Load the image
image = io.imread('House_inthe_AutumnForest.png')
image = img_as_float(image)

# Reshape the image to a 2D array of pixels
w, h, d = original_shape = tuple(image.shape)
d == 3  # Ensure it's a color image
image_array = np.reshape(image, (w * h, d))

# Define a range of number of components (Gaussians or colors)
min_components = 1
max_components = 20
n_components_range = range(min_components, max_components + 1)

# Initialize lists to store BIC scores and LA scores
bic_scores = []

# Fit Gaussian mixture models and calculate BIC for each number of components
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(image_array)
    bic_score = gmm.bic(image_array)
    bic_scores.append(bic_score)

# Find the best number of components based on BIC
best_n_components_bic = np.argmin(bic_scores) + 1

# Display color-reduced images for each value of k
plt.figure(figsize=(15, 5))
for i, n_components in enumerate(n_components_range, 1):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    image_reduced = gmm.fit_predict(image_array).reshape(image.shape[:-1])
    plt.subplot(5, 5, i)
    plt.imshow(image_reduced, cmap='hsv')
    plt.title(f'k={n_components}')
    plt.axis('off')
plt.suptitle('Color-Reduced Images')
plt.show()

# Plot EM iterations for the best number of components based on BIC
def plot_EM_iteration(gmm, X, bestK):
    gmm.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), s=5, cmap='viridis')
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='x', c='red', s=100, label='Centroids')
    plt.title('One Iteration of EM for MoG with K = ' + str(bestK))
    plt.legend()
    plt.show()
    
gmm_best = GaussianMixture(n_components=best_n_components_bic, random_state=42)
plot_EM_iteration(gmm_best, image_array, best_n_components_bic)

# Plot BIC scores
plt.plot(n_components_range, bic_scores, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('BIC Score vs Number of Components')
plt.xticks(n_components_range)
plt.axvline(x=best_n_components_bic, color='r', linestyle='--', label='Best Number of Components')
plt.legend()
plt.show()