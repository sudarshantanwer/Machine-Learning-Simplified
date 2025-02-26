import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data for apples, lemons, and cherries (weight, texture)
X_train = np.array([[150, 2], [170, 1], [140, 3],  # Apples
                    [80, 8], [85, 9], [90, 7],   # Lemons
                    [10, 10], [15, 9], [12, 10]])  # Cherries

y_train = np.array(['apple', 'apple', 'apple',    # Labels for apples
                    'lemon', 'lemon', 'lemon',   # Labels for lemons
                    'cherry', 'cherry', 'cherry'])  # Labels for cherries

# New fruit data to classify
X_new = np.array([[50, 7]])  # New fruit (Weight: 50g, Texture: 7)

# Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict the label for the new fruit
prediction = knn.predict(X_new)
print(f'Prediction: {prediction[0]}')

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(X_train[:3, 0], X_train[:3, 1], color='red', label='Apple', s=100)   # Apples in red
plt.scatter(X_train[3:6, 0], X_train[3:6, 1], color='green', label='Lemon', s=100)  # Lemons in green
plt.scatter(X_train[6:, 0], X_train[6:, 1], color='purple', label='Cherry', s=100)  # Cherries in purple
plt.scatter(X_new[:, 0], X_new[:, 1], color='blue', label='New Fruit', s=200, marker='X')  # New fruit

# Draw lines from the new fruit to its 3 nearest neighbors
nearest_neighbors = knn.kneighbors(X_new, return_distance=False)
for neighbor in nearest_neighbors[0]:
    plt.plot([X_new[0, 0], X_train[neighbor, 0]], [X_new[0, 1], X_train[neighbor, 1]], 'k--')

plt.xlabel('Weight (grams)')
plt.ylabel('Texture (Smoothness)')
plt.title('KNN Classification: Apple vs Lemon vs Cherry')
plt.legend()
plt.grid(True)
plt.show()
