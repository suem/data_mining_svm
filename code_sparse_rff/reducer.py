#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIM_ORIG = 400
DIM_TRANS = 1500


def print_array(a):
    for a_i in np.nditer(a):
        print(a_i),
    print('')


if __name__ == "__main__":
    count = 0
    weights_sum = np.zeros(DIM_TRANS)
    for line in sys.stdin:
        line = line.strip()
        weights_sum += np.fromstring(line, sep=' ')
        count += 1
    weights = weights_sum/count
    print_array(weights)
