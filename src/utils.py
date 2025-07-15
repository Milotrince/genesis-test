
from contextlib import contextmanager
from memory_profiler import memory_usage
import tracemalloc
import threading
import time
import os
import gc

@contextmanager
def timer(message="Elapsed time:"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{message} {elapsed_time:.4f} seconds")

@contextmanager
def memory_usage(message="Max memory usage (tracemalloc):"):
    gc.collect()  # Clean up before measuring
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()  # Clean up after measuring
        peak_mb = peak / (1024 * 1024)
        print(f"{message} {peak_mb:.4f} MB")


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
