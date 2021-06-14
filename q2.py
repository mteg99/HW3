import numpy as np
import pandas as pd
import multiprocessing as mp

import q2_api as q2


csv_lock = mp.Lock()
N_to_column = dict([(2, '100 Samples'), (3, '1000 Samples'), (4, '10000 Samples'), (5, '100000 Samples')])


def update_csv(N, M, path, lock):
    lock.acquire()
    df = pd.read_csv(path, index_col='Components')
    df.at[M, N_to_column[N]] += 1
    df.to_csv(path)
    lock.release()


def experiment(lock):
    for N in range(2, 6):
        D, L = q2.generate_dataset(10 ** N)
        bic = []
        log_likelihood = []
        for M in range(2, 11):
            bic.append(q2.bic(D.T, M))
            log_likelihood.append(q2.cross_validate(D.T, M, 10))
        update_csv(N, np.argmin(bic) + 2, 'bic.csv', lock)
        update_csv(N, np.argmax(log_likelihood) + 2, 'k-fold.csv', lock)


def main():
    for i in range(mp.cpu_count()):
        mp.Process(target=experiment, args=(csv_lock,)).start()


if __name__ == "__main__":
    main()

