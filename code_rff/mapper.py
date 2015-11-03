#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import time

DIM_ORIG = 400
DIM_TRANS = 3500
np.random.seed(42)
omegas = np.random.randn(DIM_TRANS, 1196)
bs = np.random.uniform(0, 2*np.pi, DIM_TRANS)


def transform(x_orig):
    x_orig = x_orig[1:]
    x_orig = np.hstack((np.sqrt(x_orig),
                        np.diff(x_orig),
                        np.gradient(x_orig)))
    return np.sqrt(2.0/DIM_TRANS)*np.cos(omegas.dot(x_orig) + bs)


def print_array(a):
    for a_i in np.nditer(a):
        print(a_i),
    print('')


def sgd(x, y, l, start_time):
    l_sqrt = np.sqrt(l)
    w = np.zeros(DIM_TRANS)
    w_avg = w
    t = 1
    while time.time() - start_time < 260:
        i = np.random.randint(x.shape[0])
        xi = x[i, :]
        if y[i]*np.dot(w, xi) < 1:
            w += y[i]*xi/(l*t)
            w *= min(1, 1/(l_sqrt*np.linalg.norm(w)))
            w_avg += w
            t += 1
    return w_avg/(t - 1)


if __name__ == '__main__':

    time_init = time.time()

    labels = []
    trans_features = []

    for line in sys.stdin:
        line = line.strip()
        label, feature = line.split(' ', 1)
        labels = np.hstack((labels, int(label)))
        feature = transform(np.fromstring(feature, sep=' '))
        trans_features.append(feature)
    trans_features = np.array(trans_features).reshape((-1, DIM_TRANS))

    lam = 1.0/trans_features.shape[0]
    weights = sgd(trans_features, labels, lam, time_init)
    print_array(weights)
