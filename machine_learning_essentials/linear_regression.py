from sklearn import model_selection
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Fetch the California housing dataset
model = datasets.fetch_california_housing()
X = model.data
y = model.target
print(f"Shape of features is {X.shape}")
print(f"Shape of labels is {y.shape} \n")

# Splitting the data into training and testing sets using an 80:20 ratio
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Splitting the data further into validation and testing sets using a 50:50 ratio
# Here, the validation set (X_val, y_val) and test set (X_test, y_test) are generated from the original test set.
# The validation set is used to tune hyperparameters and make early stopping decisions,
# while the test set is used to evaluate the model's performance after optimization.
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

class LinearRegression():
    """
    Linear Regression class for predicting target values based on input features.

    Parameters:
        n_features (int): Number of input features in the dataset.

    Attributes:
        W (ndarray): Weights for each feature.
        b (ndarray): Bias term.

    Methods:
        __init__(n_features)
            Initializes the LinearRegression class with random initial weights and bias.
        __call__(X)
            Makes predictions based on the linear regression model.
        update_parameters(W, b)
            Updates the model's weights and bias with new values.
    """
    def __init__(self, n_features):
        """
        Initializes the LinearRegression class with random initial weights and bias.

        Parameters:
            n_features (int): Number of input features in the dataset.
        """
        # By setting the random seed to a specific value (in this case, 42),
        # the same set of random numbers will be generated every time you run the code.
        np.random.seed(42)
        # Initializing random weights and bias for each feature
        self.W = np.random.randn(n_features, 1)
        self.b = np.random.randn(1)

    def __call__(self, X):
        """
        Makes predictions based on the linear regression model.

        Parameters:
            X (ndarray): Input features (samples x features).

        Returns:
            ndarray: Predicted target values (samples x 1).
        """
        ypred = np.dot(X, self.W) + self.b
        return ypred

    def update_parameters(self, W, b):
        """
        Updates the model's weights and bias with new values.

        Parameters:
            W (ndarray): New weights (features x 1).
            b (ndarray): New bias term (1).
        """
        self.W = W
        self.b = b


def mse(y_pred, y_true):
    """
    Calculates the mean squared error between predicted values and actual target.

    Parameters:
        y_pred (ndarray): Predicted values (samples x 1).
        y_true (ndarray): Actual target values (samples x 1).

    Returns:
        float: Mean squared error.
    """
    # Compute the differences between predicted and true values
    errors = y_pred - y_true
    # Square the errors element-wise to get the squared errors
    mserror = errors ** 2
    # Calculate the mean of the squared errors to get the mean squared error
    return np.mean(mserror)


def minimize_loss(X_train, y_train):
    """
    Minimizes the loss function to find the optimal weights and bias.

    Parameters:
        X_train (ndarray): Input features of the training set (samples x features).
        y_train (ndarray): Target values of the training set (samples x 1).

    Returns:
        ndarray, ndarray: Optimal weights (features x 1), Optimal bias (1).
    """
    # Adding a bias term (1) to the feature matrix for optimization
    X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # Calculating the optimal weights and bias using the training data
    optimal_w = np.matmul(
        np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
        np.matmul(X_with_bias.T, y_train),
    )
    return optimal_w[1:], optimal_w[0]


def plot_predictions(y_pred, y_true):
    """
    Plots the first few predicted and actual label values.

    Parameters:
        y_pred (ndarray): Predicted values (samples x 1).
        y_true (ndarray): Actual target values (samples x 1).
    """
    # Get the number of samples
    samples = len(y_pred)
    # Create a scatter plot to visualize predictions and true labels
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()

# Instantiate the LinearRegression class with the correct number of features
linreg = LinearRegression(X_train.shape[1])

# Call the __call__() method on the instance of the LinearRegression class to make predictions on the validation set
y_val_pred = linreg(X_val)
# Plot the scatter plot before optimizing the data
plot_predictions(y_val_pred[:10], y_test[:10])
# Calculate the mean squared error in the initial model
print("Initial Validation MSE:", mse(y_val_pred, y_test))

# Now let's optimize the data
# Calculate optimal weights and bias using the training data
weights, bias = minimize_loss(X_train, y_train)

print("\n Optimal Weights:", weights)
print("Optimal Bias:", bias)

# Updating the parameters with optimized values
linreg.update_parameters(weights, bias)

# Call the __call__() method on the instance of the LinearRegression class to make predictions on the validation set
y_val_pred_optimized = linreg(X_val)
# Calculate the Mean Squared Error (MSE) on the validation set after optimization
val_mse = mse(y_val_pred_optimized, y_val)

# Plot the scatter plot after optimizing the data
plot_predictions(y_val_pred_optimized[:10], y_val[:10])
print("\n Validation MSE after Optimization:", val_mse)