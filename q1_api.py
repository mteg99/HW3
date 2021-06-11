import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import stats

# Distribution parameters
C = 4
mu0 = np.array([[1, 1, 1], [1, 1, -1]])
mu1 = np.array([[1, -1, 1], [1, -1, -1]])
mu2 = np.array([[-1, 1, 1], [-1, 1, -1]])
mu3 = np.array([[-1, -1, 1], [1, -1, -1]])
mu = np.array([mu0, mu1, mu2, mu3])
sigma0 = np.zeros(shape=(2, 3, 3))
sigma0[0] = 0.1 * np.identity(3)
sigma0[1] = 0.2 * np.identity(3)
sigma1 = np.zeros(shape=(2, 3, 3))
sigma1[0] = 0.1 * np.identity(3)
sigma1[1] = 0.2 * np.identity(3)
sigma2 = np.zeros(shape=(2, 3, 3))
sigma2[0] = 0.1 * np.identity(3)
sigma2[1] = 0.2 * np.identity(3)
sigma3 = np.zeros(shape=(2, 3, 3))
sigma3[0] = 0.1 * np.identity(3)
sigma3[1] = 0.2 * np.identity(3)
sigma = np.array([sigma0, sigma1, sigma2, sigma3])


def generate_dataset(n):
    data = np.zeros(shape=(3, n))
    labels = np.zeros(shape=n, dtype=int)
    for i in range(n):
        l = random.uniform(0, 1)
        if l < 0.25:
            labels[i] = 0
        elif l < 0.5:
            labels[i] = 1
        elif l < 0.75:
            labels[i] = 2
        else:
            labels[i] = 3
        if random.uniform(0, 1) < 0.5:
            data[:, i] = np.random.multivariate_normal(mean=mu[labels[i]][0], cov=sigma[labels[i]][0]).T
        else:
            data[:, i] = np.random.multivariate_normal(mean=mu[labels[i]][1], cov=sigma[labels[i]][1]).T
    return data, labels


def plot_dataset(data, labels, title):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    L0 = data[:, [i == 0 for i in labels]]
    L1 = data[:, [i == 1 for i in labels]]
    L2 = data[:, [i == 2 for i in labels]]
    L3 = data[:, [i == 3 for i in labels]]
    ax.scatter(L0[0], L0[1], L0[2], color='red')
    ax.scatter(L1[0], L1[1], L1[2], color='yellow')
    ax.scatter(L2[0], L2[1], L2[2], color='green')
    ax.scatter(L3[0], L3[1], L3[2], color='blue')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.title(title)
    plt.show()


def optimal_pr_error(data, labels):
    N = labels.size
    p = np.zeros(shape=(C, N))
    d = np.zeros(shape=N, dtype=int)
    for n in range(labels.size):
        for c in range(C):
            p[c][n] = 0.5 * stats.multivariate_normal.pdf(data[:, n], mu[c][0], sigma[c][0]) + \
                      0.5 * stats.multivariate_normal.pdf(data[:, n], mu[c][1], sigma[c][1])
        d[n] = np.argmax(p[:, n])
    pr_error = np.count_nonzero([d[i] != labels[i] for i in range(N)]) / N
    return pr_error
