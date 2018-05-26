from ceres_python import optimize
from functools import partial
import numpy as np

def func(ps, target=None):
    assert(ps.shape == target.shape)
    return np.sum((ps-target)**2)

def grad(ps, target=None):
    return 2 * (ps - target)

x0 = np.array(range(1, 6), dtype=np.float)
target = np.array([3.725] * 5)

print(optimize(partial(func, target=target), partial(grad, target=target), x0))
