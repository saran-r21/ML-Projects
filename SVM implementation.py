import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets (70:30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Varying values of the penalty term C
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
precisions = []
recalls = []

for C in C_values:
    # Create and train SVM model with a linear kernel and varying C
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm.predict(X_test)
    
    # Calculate precision and recall and store them
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    
    precisions.append(precision)
    recalls.append(recall)

# Plot the graph of C versus precision and recall
plt.figure(figsize=(10, 6))
plt.plot(C_values, precisions, marker='o', label='Precision')
plt.plot(C_values, recalls, marker='o', label='Recall')
plt.xscale('log')
plt.xlabel('Penalty Term C')
plt.ylabel('Score')
plt.title('SVM Precision and Recall vs. Penalty Term C (Breast Cancer Dataset)')
plt.legend()
plt.grid(True)
plt.show()
