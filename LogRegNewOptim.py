import warnings

import numpy as np
# data manipulation
import pandas as pd

# makes the random numbers predictable
# (pseudo-)random numbers work by starting with a number (the seed),
# multiplying it by a large number, then taking modulo of that product.
# The resulting number is then used as the seed to generate the next "random" number.
# When you set the seed (every time), it does the same thing every time, giving you the same numbers.
# good for reproducing results for debugging


np.random.seed(0)  # set the seed


# outputs probability between 0 and 1, used to help define our logistic regression curve
def sigmoid(x):
    """Sigmoid function of x."""
    return 1 / (1 + np.exp(-x))


# Step 1 - Define model parameters (hyperparameters)

# algorithm settings
# the minimum threshold for the difference between the predicted output and the actual output
# this tells our model when to stop learning, when our prediction capability is good enough
tol = 1e-8  # convergence tolerance

lam = None  # l2-regularization
# how long to train for?
max_iter = 20  # maximum allowed iterations

# data creation settings
# Covariance measures how two variables move together.
# It measures whether the two move in the same direction (a positive covariance)
# or in opposite directions (a negative covariance).
r = 0.95  # covariance between x and z
n = 1000  # number of observations (size of dataset to generate)
sigma = 1  # variance of noise - how spread out is the data?

# model settings
beta_x, beta_z, beta_v = -4, .9, 1  # true beta coefficients
var_x, var_z, var_v = 1, 1, 4  # variances of inputs

# the model specification you want to fit
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

# Step 2 - Generate and organize our data

# The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal
# distribution to higher dimensions. Such a distribution is specified by its mean and covariance matrix.
# so we generate values input values - (x, v, z) using normal distributions

# A probability distribution is a function that provides us the probabilities of all
# possible outcomes of a stochastic process.

# lets keep x and z closely related (height and weight)
x, z = np.random.multivariate_normal([0, 0], [[var_x, r], [r, var_z]], n).T
# blood presure
v = np.random.normal(0, var_v, n) ** 3

# create a pandas dataframe (easily parseable object for manipulation)
A = pd.DataFrame({'x': x, 'z': z, 'v': v})
# compute the log odds for our 3 independent variables
# using the sigmoid function
A['log_odds'] = sigmoid(A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) + sigma * np.random.normal(0, 1, n))

# compute the probability sample from binomial distribution
# A binomial random variable is the number of successes x has in n repeated trials of a binomial experiment.
# The probability distribution of a binomial random variable is called a binomial distribution.
A['y'] = [np.random.binomial(1, p) for p in A.log_odds]

# create a dataframe that encompasses our input data, model formula, and outputs
y, X = dmatrices(formula, A, return_type='dataframe')

# print it
X.head(100)


# like dividing by zero (Wtff omgggggg universe collapses)
def catch_singularity(f):
    '''Silences LinAlg Errors and throws a warning instead.'''

    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]

    return silencer


@catch_singularity
def newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''

    # how to compute inverse? http://www.mathwarehouse.com/algebra/matrix/images/square-matrix/inverse-matrix.gif

    # compute necessary objects
    # create probability matrix, miniminum 2 dimensions, tranpose (flip it)
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    # create weight matrix from it
    W = np.diag((p * (1 - p))[:, 0])
    # derive the hessian
    hessian = X.T.dot(W).dot(X)
    # derive the gradient
    grad = X.T.dot(y - p)

    # regularization step (avoiding overfitting)
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    # update our
    beta = curr + step

    return beta


def check_coefs_convergence(beta_old, beta_new, tol, iters):
    '''Checks whether the coefficients have converged in the l-infinity norm.
    Returns True if they have converged, False otherwise.'''
    # calculate the change in the coefficients
    coef_change = np.abs(beta_old - beta_new)

    # if change hasn't reached the threshold and we have more iterations to go, keep training
    return not (np.any(coef_change > tol) & (iters < max_iter))
