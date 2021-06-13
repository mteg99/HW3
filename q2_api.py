import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture

# Distribution parameters
C = 5
mu0 = np.array([-4, 0])
mu1 = np.array([-2, 0])
mu2 = np.array([0, 0])
mu3 = np.array([2, 0])
mu4 = np.array([4, 0])
mu = np.array([mu0, mu1, mu2, mu3, mu4])
sigma0 = 0.1 * np.identity(2)
sigma1 = 0.2 * np.identity(2)
sigma2 = 0.3 * np.identity(2)
sigma3 = 0.4 * np.identity(2)
sigma4 = 0.5 * np.identity(2)
sigma = np.array([sigma0, sigma1, sigma2, sigma3, sigma4])


def generate_dataset(n):
    data = np.zeros(shape=(2, n))
    labels = np.zeros(shape=n, dtype=int)
    for i in range(n):
        l = random.uniform(0, 1)
        if l < 0.20:
            labels[i] = 0
        elif l < 0.40:
            labels[i] = 1
        elif l < 0.60:
            labels[i] = 2
        elif l < 0.80:
            labels[i] = 3
        else:
            labels[i] = 4
        data[:, i] = np.random.multivariate_normal(mean=mu[labels[i]], cov=sigma[labels[i]]).T
    return data, labels


def plot_dataset(data, labels, title):
    fig, ax = plt.subplots()
    L0 = data[:, [i == 0 for i in labels]]
    L1 = data[:, [i == 1 for i in labels]]
    L2 = data[:, [i == 2 for i in labels]]
    L3 = data[:, [i == 3 for i in labels]]
    L4 = data[:, [i == 4 for i in labels]]
    ax.scatter(L0[0], L0[1], color='red')
    ax.scatter(L1[0], L1[1], color='orange')
    ax.scatter(L2[0], L2[1], color='yellow')
    ax.scatter(L3[0], L3[1], color='green')
    ax.scatter(L4[0], L4[1], color='blue')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.title(title)
    plt.show()


def cross_validate(D, M, K):
    log_likelihood = []
    kf = KFold(n_splits=K)
    for train_index, test_index in kf.split(D):
        D_train = np.array([D[i] for i in train_index])
        D_test = np.array([D[i] for i in test_index])
        gmm = GaussianMixture(n_components=M).fit(D_train)
        log_likelihood.append(np.sum(gmm.score_samples(D_test)))
    avg_log_likelihood = np.sum(log_likelihood) / K
    return avg_log_likelihood


def bic(D, M):
    n = D.shape[1]
    k = M * (2 + 4)  # number of components times 2 means plus 4 covariance values
    gmm = GaussianMixture(n_components=M).fit(D)
    return k * np.log(n) - 2 * np.sum(gmm.score_samples(D))
