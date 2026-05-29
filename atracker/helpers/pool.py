#! /usr/bin/env python

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from pythutils.sysutils import lineprint
from .data import notebook


def run_pool(worker, items, pools=1, mode="thread", label="task"):
    """
    Run worker over items sequentially (pools=1) or in a pool (pools>1).

    Parameters
    ----------
    worker : callable
        For mode='process', called as worker(*item) when item is a tuple,
        worker(item) otherwise. For mode='thread', always called as worker(item).
    items : list
        Items to process.
    pools : int
        Number of parallel workers. 1 = sequential.
    mode : str
        'thread' uses ThreadPoolExecutor; 'process' uses multiprocessing.Pool.
    label : str
        Used in progress and error messages.

    Returns
    -------
    bool
        True on success, False if interrupted or fatally failed.
    """
    if not items:
        return True

    if pools < 2:
        for item in items:
            try:
                if mode == "process" and isinstance(item, tuple):
                    worker(*item)
                else:
                    worker(item)
            except KeyboardInterrupt:
                lineprint(f"\nUser terminated {label}..")
                return False
        lineprint(f"{label.capitalize()} completed..")
        return True

    if notebook():
        lineprint(f"Pooled {label} can only be run from the terminal, exiting..")
        return False

    if mode == "thread":
        try:
            with ThreadPoolExecutor(max_workers=pools) as executor:
                executor.map(worker, items)
            lineprint(f"{label.capitalize()} completed..")
            return True
        except KeyboardInterrupt:
            lineprint(f"\nUser terminated {label} pool..")
            return False

    # mode == "process"
    pool = multiprocessing.Pool(min(pools, len(items)))
    try:
        args_list = [item if isinstance(item, tuple) else (item,) for item in items]
        futures = [pool.apply_async(worker, args) for args in args_list]
        for f in futures:
            f.get()
        pool.close()
        lineprint(f"{label.capitalize()} completed..")
        return True
    except KeyboardInterrupt:
        lineprint(f"\nUser terminated {label} pool..")
        pool.terminate()
        return False
    except Exception as e:
        lineprint(f"Error in {label} pool: {type(e).__name__}: {e}, terminating")
        pool.terminate()
        return False
    finally:
        pool.join()
