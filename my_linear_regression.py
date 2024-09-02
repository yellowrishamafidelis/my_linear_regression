import numpy as np
import sys

# 1. Linear hypothesis function
def h(x, theta):
    return np.dot(x, theta).reshape(-1, 1)

# 2. Mean squared error function
def mean_squared_error(y_predicted, y_label):
    return np.mean((y_predicted - y_label) ** 2)

# 3. Bias column function
def bias_column(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

# 4. Least squares regression class
class LeastSquaresRegression():
    def __init__(self):
        self.theta_ = None

    def fit(self, X, y):
        X_transpose = X.T
        self.theta_ = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    def predict(self, X):
        return h(X, self.theta_)

# 5. Gradient Descent Optimizer class
class GradientDescentOptimizer():
    def __init__(self, f, fprime, start, learning_rate=0.1):
        self.f_ = f
        self.fprime_ = fprime
        self.current_ = start.reshape(-1, 1)
        self.learning_rate_ = learning_rate
        self.history_ = [start]

    def step(self):
        gradient = self.fprime_(self.current_)
        self.current_ = self.current_ - self.learning_rate_ * gradient
        # print(self.current_, file=sys.stderr)
        self.history_.append(self.current_.copy())

    def optimize(self, iterations=100):
        for _ in range(iterations):
            self.step()

    def getCurrentValue(self):
        return self.current_

    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))

# 6. Gradient Descent example functions
def f(x):
    target = np.array([2, 6])
    return 3 + np.dot((x - target).T, (x - target))

def fprime(x):
    target = np.array([2, 6])
    return 2 * (x - target)
