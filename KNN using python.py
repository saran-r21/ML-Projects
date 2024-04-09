from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature values (optional but recommended for kNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

k_values = range(1, 50)  # Try different values of k from 1 to 50
precisions = []

for k in k_values:
    # Create and train kNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate precision and store it
    precision = precision_score(y_test, y_pred, average='weighted')
    precisions.append(precision)

# Plot k versus precision
plt.plot(k_values, precisions, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Precision')
plt.title('kNN Precision vs. k (Iris Dataset)')
plt.grid(True)
plt.show()
