import matplotlib.pyplot as plt
import numpy as np


NUM_SAMPLE = 4
def func(x):
    return -x*np.sin(x)
def kernel(a, b):
    return np.exp(-0.5*np.square(np.subtract.outer(a, b)))


x_space = np.arange(-2*np.pi, 2*np.pi, 0.01)
true_y = func(x_space)

train_idx = np.random.randint(len(x_space), size=NUM_SAMPLE)
train_x = x_space[train_idx]
train_y = true_y[train_idx]

test_x = x_space

# [  y_test ] ~ N ( 0, [ c   k ] )
# [ y_train ]          [ k.T C ]
C = kernel(train_x, train_x)
C_inv = np.linalg.inv(C)
k = kernel(test_x, train_x)
c = kernel(test_x, test_x)

mu = k@C_inv@train_y
sigma = c - k@C_inv@k.T
test_pred = np.random.multivariate_normal(mu, sigma)
pointwise_std = np.sqrt(np.diag(sigma))

plt.figure()
plt.plot(x_space, true_y)
plt.scatter(train_x, train_y, marker='x')
plt.plot(test_x, test_pred)
plt.fill_between(x_space, mu - 2*pointwise_std, mu + 2*pointwise_std, alpha=0.3)
plt.show()
