import numpy as np
import multiprocessing
from joblib import Parallel, delayed

import q2_api as q2


def experiment_loop():
    for N in range(2, 6):
        D, L = q2.generate_dataset(10 ** N)
        bic = []
        log_likelihood = []
        for M in range(2, 10):
            bic.append(q2.bic(D.T, M))
            log_likelihood.append(q2.cross_validate(D.T, M, 10))
        print('10^' + str(N) + ' Samples:')
        print('BIC: ' + str(np.argmin(bic) + 2))
        print('K-fold: ' + str(np.argmax(log_likelihood) + 2))


Parallel(n_jobs=multiprocessing.cpu_count())(delayed(experiment_loop)() for i in range(100))
