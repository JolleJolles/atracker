"""Shared batch execution with visible per-item failures."""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging import lineprint
from .data import notebook


class BatchInterrupted(Exception):
    """Serializable cancellation signal for process workers."""


def run_pool(worker, items, pools=1, mode="thread", label="task"):
    """Run every item, report failures and return False on any failure/interruption.

    Process mode unpacks tuple items; thread mode passes each item unchanged.
    Ordinary per-file errors do not prevent other files running. On interruption,
    pending thread tasks are cancelled and active threads finish before returning;
    process workers are terminated and joined. Process pools require a script.
    """
    if mode not in ("thread", "process") or not isinstance(pools, int) or pools < 1:
        raise ValueError("Use mode='thread' or 'process' and pools >= 1")
    items = list(items)
    if not items:
        return True
    failures = []

    def failed(item, error):
        failures.append(item)
        lineprint(f"{label.capitalize()} failed for {item!r}: {type(error).__name__}: {error}")

    def summary():
        lineprint(f"{label.capitalize()}: {len(items)-len(failures)}/{len(items)} tasks succeeded; {len(failures)} failed.")
        return not failures

    if pools == 1:
        for item in items:
            try:
                worker(*item) if mode == "process" and isinstance(item, tuple) else worker(item)
            except (KeyboardInterrupt, BatchInterrupted):
                lineprint(f"User interrupted {label}; remaining tasks were not started.")
                return False
            except Exception as exc:
                failed(item, exc)
        return summary()

    if mode == "thread":
        executor = ThreadPoolExecutor(max_workers=min(pools, len(items)))
        futures = {executor.submit(worker, item): item for item in items}
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except (KeyboardInterrupt, BatchInterrupted):
                    raise
                except Exception as exc:
                    failed(futures[future], exc)
        except (KeyboardInterrupt, BatchInterrupted):
            for future in futures:
                future.cancel()
            lineprint(f"User interrupted {label}; pending tasks cancelled, waiting for active files to finish.")
            return False
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return summary()

    if notebook():
        lineprint(f"Process-pooled {label} requires a terminal script with a main guard.")
        return False
    pool = multiprocessing.Pool(min(pools, len(items)))
    try:
        futures = [(item, pool.apply_async(worker, item if isinstance(item, tuple) else (item,))) for item in items]
        for item, future in futures:
            try:
                future.get()
            except (KeyboardInterrupt, BatchInterrupted):
                raise
            except Exception as exc:
                failed(item, exc)
        pool.close()
        return summary()
    except (KeyboardInterrupt, BatchInterrupted):
        lineprint(f"User interrupted {label}; terminating process workers.")
        pool.terminate()
        return False
    except Exception as exc:
        lineprint(f"{label.capitalize()} pool failed: {type(exc).__name__}: {exc}")
        pool.terminate()
        return False
    finally:
        pool.join()
