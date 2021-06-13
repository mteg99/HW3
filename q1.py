import numpy as np
import multiprocessing
from joblib import Parallel, delayed

import q1_api as q1

# Generate datasets
D100_train, L100_train = q1.generate_dataset(100)
D200_train, L200_train = q1.generate_dataset(200)
D500_train, L500_train = q1.generate_dataset(500)
D1k_train, L1k_train = q1.generate_dataset(1000)
D2k_train, L2k_train = q1.generate_dataset(2000)
D5k_train, L5k_train = q1.generate_dataset(5000)
D_train = [D100_train, D200_train, D500_train, D1k_train, D2k_train, D5k_train]
L_train = [L100_train, L200_train, L500_train, L1k_train, L2k_train, L5k_train]
D100k_test, L100k_test = q1.generate_dataset(100000)

# Plot datasets
# q1.plot_dataset(D100_train, L100_train, '100 Training Samples')
# q1.plot_dataset(D200_train, L200_train, '200 Training Samples')
# q1.plot_dataset(D500_train, L500_train, '500 Training Samples')
# q1.plot_dataset(D1k_train, L1k_train, '1000 Training Samples')
# q1.plot_dataset(D2k_train, L2k_train, '2000 Training Samples')
# q1.plot_dataset(D5k_train, L5k_train, '5000 Training Samples')
# q1.plot_dataset(D100k_test, L100k_test, '100000 Training Samples')

# Determine optimal P
P = []
for n in range(6):
    pr_error = []
    for p in range(1, 17):
        print('D = ' + str(n) + ' P = ' + str(p))
        pr_error.append(q1.cross_validate(D_train[n].T, L_train[n], K=5, P=p))
    P.append(np.argmin(pr_error) + 1)
print('Optimal P:')
print(P)

# Train MLP models with optimal P
pr_error = []
for n in range(6):
    pr_error.append(q1.evaluate_MLP(D_train[n].T, L_train[n], D100k_test.T, L100k_test, P[n]))
print('Pr(error):')
print(pr_error)

# Optimal Classifier
print('Optimal Pr(error):')
print(q1.optimal_pr_error(D100k_test, L100k_test))
