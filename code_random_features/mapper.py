#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIM_ORIG = 400
DIM_TRANS = 1000
np.random.seed(42)
dirs = np.random.randn(DIM_TRANS, DIM_ORIG)
bs = np.random.uniform(0, 2*np.pi, DIM_TRANS)


def rand_feature(x_orig):
    return np.sqrt(2/DIM_TRANS) * dirs.dot(x_orig)+bs


def transform(x_orig):
    return rand_feature(x_orig)


def print_array(a):
    for a_i in np.nditer(a):
        print(a_i),
    print('')


def sgd_step(t, w_t, x, y, l, l_sqrt):
    w_next = w_t
    if y*np.dot(w_t, x) < 1:
        w = w_t + y*x/(t*l)
        w_next = w * min(1, 1/(l_sqrt*np.linalg.norm(w)))
    return w_next


if __name__ == '__main__':

    lam = 0.0001
    lam_sqrt = np.sqrt(lam)
    t = 1
    w = np.zeros(DIM_TRANS)

    for line in sys.stdin:
        line = line.strip()
        label, features = line.split(' ', 1)
        y = int(label)
        x = transform(np.fromstring(features, sep=' '))
        w = sgd_step(t, w, x, y, lam, lam_sqrt)
        t += 1

    w_averaged = w / t
    print_array(w_averaged)