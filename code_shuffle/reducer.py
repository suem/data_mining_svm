#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

DIMENSION = 400  # Dimension of the original data.
DIMENSION_T = 401  # Dimension of the original data.
CLASSES = (-1, +1)  # The classes that we are trying to predict.


def print_vector(v):
    for x_i in np.nditer(v):
        print(x_i),
    print('')


if __name__ == "__main__":
    w_count = 0
    w_total = np.zeros(DIMENSION_T)
    for line in sys.stdin:
        line = line.strip()
        (iteration_id, w_string) = line.split("\t", 1)
        w = np.fromstring(w_string, sep=' ')
        w_total += w
        w_count += 1
    w_averaged = w_total / w_count
    print_vector(w_averaged)

