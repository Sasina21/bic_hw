import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load the image
image = cv2.imread('House_inthe_AutumnForest.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flatten the image to 2D array
pixels = np.reshape(image, (-1, 3))

# Initialize lists to store BIC values and color-reduced images
bic_values = []
color_reduced_images = []

# Loop through different values of k
for k in range(1, 21):
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm.fit(pixels)
    
    # Compute BIC
    bic = gmm.bic(pixels)
    bic_values.append(bic)
    
    # Generate color-reduced image
    labels = gmm.predict(pixels)
    clustered_pixels = gmm.means_[labels]
    clustered_image = np.reshape(clustered_pixels, image.shape)
    color_reduced_images.append(clustered_image)

# Determine the best value of k based on minimum BIC
best_k = np.argmin(bic_values) + 1

# Plot BIC graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), bic_values, marker='o')
plt.xlabel('Number of Gaussians (k)')
plt.ylabel('BIC')
plt.title('BIC vs Number of Gaussians')
plt.grid(True)
plt.show()

# Plot EM-MOG graph for best k
plt.figure(figsize=(10, 5))
gmm_best_k = GaussianMixture(n_components=best_k, random_state=0)
log_likelihoods = gmm_best_k.fit(pixels).score_samples(pixels)
plt.plot(np.arange(len(log_likelihoods)), log_likelihoods, label='Log Likelihood')
plt.xlabel('Number of Iterations')
plt.ylabel('Log Likelihood')
plt.title(f'EM-MOG with {best_k} Gaussians')
plt.legend()
plt.grid(True)
plt.show()

# Plot color-reduced images for each k
plt.figure(figsize=(20, 10))
for i in range(1, 21):
    plt.subplot(4, 5, i)
    plt.imshow(color_reduced_images[i-1])
    plt.title(f'k={i}')
    plt.axis('off')
plt.suptitle('Color-Reduced Images for Different Values of k', fontsize=16)
plt.show()