#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400  # Dimension of the original data.
DIMENSION_T = 401  # Dimension of the original data.
CLASSES = (-1, +1)  # The classes that we are trying to predict.


def transform(x_orig):
    return np.append(x_orig, 1)


def print_vector(v):
    for x_i in np.nditer(v):
        print(x_i),
    print('')


def step(t, lam_sqrt, w_prev, x, y):
    w_next = w_prev
    eta = 1/np.sqrt(t)
    if y*x.dot(w_prev) < 1:
        w_prim = w_prev+eta*y*x
        w_next = w_prim * min(1, 1/(np.linalg.norm(w_prim)*lam_sqrt))
    return w_next


def pegasus_step(t, lam, lam_sqrt, w_t, x, y):
    eta = 1 / (lam * t)
    w_prim = w_t - eta * lam * w_t
    if y * w_t.dot(x) < 1:
        w_prim += eta * y * x
    return w_prim * min(1, 1/(lam_sqrt*np.linalg.norm(w_prim)))


if __name__ == "__main__":
    w_t = np.zeros(DIMENSION_T)
    w_total = w_t
    t = 1
    lam = 1
    lam_sqrt = np.sqrt(lam)
    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)
        y = int(label)
        # w_t = pegasus_step(t, lam, lam_sqrt, w_t, x, y)
        w_t = step(t, lam_sqrt, w_t, x, y) # this performs very bad
        t += 1
        w_total += w_t
    print('%d\t' % t),
    print_vector(w_total)