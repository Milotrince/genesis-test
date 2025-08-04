import os
import time
from contextlib import contextmanager

import numpy as np
import torch

from genesis.utils.misc import tensor_to_array


@contextmanager
def timer(message="Elapsed time:"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{message} {elapsed_time:.4f} seconds")


# @contextmanager
# def memory_usage(message="Max memory usage (tracemalloc):"):
#     gc.collect()  # Clean up before measuring
#     tracemalloc.start()
#     try:
#         yield
#     finally:
#         current, peak = tracemalloc.get_traced_memory()
#         tracemalloc.stop()
#         gc.collect()  # Clean up after measuring
#         peak_mb = peak / (1024 * 1024)
#         print(f"{message} {peak_mb:.4f} MB")


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=""):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    if isinstance(actual, torch.Tensor):
        actual = tensor_to_array(actual)
    actual = np.asanyarray(actual)
    if isinstance(desired, torch.Tensor):
        desired = tensor_to_array(desired)
    desired = np.asanyarray(desired)

    if all(e.size == 0 for e in (actual, desired)):
        return

    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol, err_msg=err_msg)


def assert_array_equal(actual, desired, *, err_msg=""):
    assert_allclose(actual, desired, atol=0.0, rtol=0.0, err_msg=err_msg)


def get_next_filename(path):
    """
    Generate auto-incrementing filename.
    Example: out_##.mp4 -> out_01.mp4, out_02.mp4, ...
    """
    counter = 1
    pad = path.count("#") or 2
    ext = path.split(".")[-1]
    path = path.rsplit(".", 1)[0]
    while True:
        filename = f"{path.replace('#' * pad, str(counter).zfill(pad))}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1
