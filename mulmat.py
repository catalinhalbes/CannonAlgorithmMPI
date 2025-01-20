#!/usr/bin/env python3
from runutils import loadmat, writemat
import numpy as np
import sys

from time import perf_counter

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} [mat1] [mat2] [rez]")
        exit(0)
    
    before_read = perf_counter()
    mat1 = loadmat(sys.argv[1])
    mat2 = loadmat(sys.argv[2])
    before_dot = perf_counter()
    res = np.dot(mat1, mat2)
    after_dot = perf_counter()
    writemat(sys.argv[3], res)
    after_read = perf_counter()

    read_time = (before_dot - before_read) * 1000
    print(f"{read_time:.6f}")
    dot_time = (after_dot - before_dot) * 1000
    print(f"{dot_time:.6f}")
    write_time = (after_read - after_dot) * 1000
    print(f"{write_time:.6f}")
