#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np


def transform(x_orig):
    return np.append(x_orig, 1)


if __name__ == "__main__":
    rounds = 5
    for line in sys.stdin:
        line = line.strip()
        for i in range(rounds):
            rand = np.random.randint(sys.maxint)
            print("%d\t%s" % (rand, line))
