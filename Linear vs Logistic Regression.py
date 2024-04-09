import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Generate a synthetic dataset with noise for both regression and classification
np.random.seed(0)
X = np.linspace(0, 10, 100)
y_regression = 3 * X + 2 + np.random.normal(0, 2, 100)  # Linear relationship with noise
y_classification = (X + np.random.normal(0, 1, 100) > 5).astype(int)  # Binary classification with noise

# Split the data into training and testing sets
X_train, X_test, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.3, random_state=42)
X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification, test_size=0.3, random_state=42)

# Model 2a - Linear Regression
# Fit a linear regression model
lr = LinearRegression()
lr.fit(X_train.reshape(-1, 1), y_train_regression)

# Model 2b - Logistic Regression
# Fit a logistic regression model
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train.reshape(-1, 1), y_train_classification)

# Model 2c - Learning Rate α versus RMSE (for Linear Regression with Gradient Descent)
learning_rates = np.linspace(0.0001, 0.01, 100)  # Reduced learning rate
rmse_values = []

for alpha in learning_rates:
    # Initialize coefficients randomly
    theta = np.random.randn(2)
    
    # Implement gradient descent manually with the specified learning rate (alpha)
    n_iterations = 1000
    m = len(X_train)

    for iteration in range(n_iterations):
        X_b = np.column_stack((np.ones_like(X_train), X_train))  # Add a bias term
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y_train_regression)
        
        # Update the coefficients while checking for NaN or overflow
        if np.isnan(gradients).any() or np.isinf(gradients).any():
            print(f"NaN or Inf encountered at iteration {iteration} with learning rate {alpha}")
            break
        
        theta = theta - alpha * gradients
    
    # Make predictions on the test set
    X_test_b = np.column_stack((np.ones_like(X_test), X_test))
    y_pred = X_test_b.dot(theta)
    
    # Calculate RMSE and store it
    rmse = np.sqrt(mean_squared_error(y_test_regression, y_pred))
    rmse_values.append(rmse)

# Plot the results
plt.figure(figsize=(14, 6))

# Model 2a - Linear Regression
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train_regression, label='Training Data')
plt.plot(X_train, lr.predict(X_train.reshape(-1, 1)), color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

# Model 2b - Logistic Regression
plt.subplot(2, 2, 2)
plt.scatter(X_train, y_train_classification, label='Training Data')
plt.plot(X, logistic_reg.predict_proba(X.reshape(-1, 1))[:, 1], color='red', label='Logistic Regression')
plt.xlabel('X')
plt.ylabel('y (Probability)')
plt.title('Logistic Regression')
plt.legend()

# Model 2c - Learning Rate α versus RMSE (for Linear Regression)
plt.subplot(2, 2, 3)
plt.plot(learning_rates, rmse_values, marker='o')
plt.xlabel('Learning Rate (α)')
plt.ylabel('RMSE (Root Mean Squared Error)')
plt.title('Learning Rate vs. RMSE (Linear Regression)')

plt.tight_layout()
plt.show()
