import math
import numpy as np


def distance(p1, p2, exp=2):
    s = 200
    d = math.sqrt(sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))
    a = math.exp(-abs(d/s)**exp)
    return np.float32(a)
