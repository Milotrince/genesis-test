
import os
import time
from contextlib import contextmanager

@contextmanager
def timer():
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")


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
        filename = f"{path.replace('#'*pad, str(counter).zfill(pad))}.{ext}"
        if not os.path.exists(filename):
            return filename
        counter += 1

import numpy as np
def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if all(isinstance(e, np.ndarray) and e.size == 0 for e in (actual, desired)):
        return
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0
    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol, err_msg=err_msg)
