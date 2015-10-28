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


if __name__ == "__main__":
    lam = 0.5
    lam_sqrt = np.sqrt(lam)

    n_points = 100000
    # n_points = 1000
    points = np.zeros([n_points, DIMENSION_T + 1])
    points_counter = 0

    # n_iters = 1
    n_iters = 100
    w_t_iters = [np.zeros(DIMENSION_T)] * n_iters
    w_total_iters = [np.zeros(DIMENSION_T)] * n_iters
    t_iters = [1] * n_iters

    for line in sys.stdin:
        line = line.strip()
        v = transform(np.fromstring(line, sep=' '))
        points[points_counter] = v
        points_counter += 1
        if points_counter == n_points:
            for j in range(n_iters):  # do n_iters iterations
                prev = np.zeros(DIMENSION_T)
                for i in np.random.permutation(points_counter): #iterate over random permutation
                    point = points[i]
                    y = int(point[0])
                    x = point[1:]
                    w_t_iters[j] = step(t_iters[j], lam_sqrt, w_t_iters[j], x, y)
                    t_iters[j] += 1
                    w_total_iters[j] += w_t_iters[j]

            # read the next set of points
            points_counter = 0

    # process remaining set of points
    if points_counter > 0:
        for j in range(n_iters):
            for i in np.random.permutation(points_counter): #iterate over random permutation
                point = points[i]
                y = int(point[0])
                x = point[1:]
                w_t_iters[j] = step(t_iters[j], lam_sqrt, w_t_iters[j], x, y)
                t_iters[j] += 1
                w_total_iters[j] += w_t_iters[j]

    # print out all averaged w's for all iterations
    for j in range(n_iters):
        print('%d\t' % j),
        w_averaged = w_total_iters[j] / t_iters[j]
        print_vector(w_averaged)
        # print_vector(w_t_iters[j])
