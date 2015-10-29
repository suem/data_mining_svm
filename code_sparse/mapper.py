#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
from scipy import sparse
import time

DIM_ORIG = 400
DIM_TRANS = 1698


def transform(x_orig):
    return np.hstack((np.sqrt(x_orig),
                      np.diff(x_orig),
                      np.gradient(x_orig),
                      np.cumsum(x_orig),
                      np.percentile(x_orig, range(1, 100))))


def print_array(a):
    for a_i in np.nditer(a):
        print(a_i),
    print('')


def sgd(x, y, l, start_time):
    l_sqrt = np.sqrt(l)
    w = np.zeros(DIM_TRANS)
    w_avg = w
    t = 1
    while time.time() - start_time < 250:
        i = np.random.randint(x.shape[0])
        xi = x[i, :].toarray().flatten()
        if y[i]*np.dot(w, xi) < 1:
            w += y[i]*xi/(l*t)
            w *= min(1, 1/(l_sqrt*np.linalg.norm(w)))
            w_avg += w
            t += 1
    return w_avg/(t - 1)


if __name__ == '__main__':

    time_init = time.time()

    labels = []
    data = []
    rows = []
    cols = []
    row = 0

    for line in sys.stdin:
        line = line.strip()
        label, features = line.split(' ', 1)
        labels = np.hstack((labels, int(label)))
        features = transform(np.fromstring(features, sep=' '))
        for col in range(DIM_TRANS):
            if col:
                data.append(features[col])
                rows.append(row)
                cols.append(col)
        row += 1

    data = np.array(data, copy=False)
    rows = np.array(rows, dtype=np.float, copy=False)
    cols = np.array(cols, dtype=np.float, copy=False)
    m = sparse.coo_matrix((data, (rows, cols)), shape=(row, DIM_TRANS))
    m = sparse.csr_matrix(m)

    lam = 1.0/row
    weights = sgd(m, labels, lam, time_init)
    print_array(weights)
