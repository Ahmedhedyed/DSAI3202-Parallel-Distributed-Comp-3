import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np 

def square(n):
    return n * n



# 1. Define a function to compute the square of a number.
def square(n):
    return n * n

def worker(idx, n, results):
    """Worker function defined at the top-level so it can be pickled."""
    results[idx] = square(n)

def chunk_list(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def chunked_square(chunk):
    # Process a list (chunk) of numbers
    return [n * n for n in chunk]

# 2. Create a list of numbers
def create_numbers(count):
    numbers = np.arange(count)

    return numbers

# 3.a. Sequential execution
def sequential_square(numbers):
    results = []
    for n in numbers:
        results.append(square(n))
    return results

# 3.b. Multiprocessing: One process per number (Not recommended for very large lists)
def mp_individual_process(numbers):
    processes = []
    manager = mp.Manager()
    results = manager.list([None] * len(numbers))

    for idx, n in enumerate(numbers):
        p = mp.Process(target=worker, args=(idx, n,results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    return list(results)

# 3.c. Multiprocessing Pool using map() and apply()
def mp_pool_map(numbers):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(square, numbers)
    return results

def mp_pool_apply(numbers):
    # Here we simulate apply by calling apply_async in a loop and then collecting the results.
    with mp.Pool(processes=mp.cpu_count()) as pool:
        async_results = [pool.apply_async(square, args=(n,)) for n in numbers]
        results = [res.get() for res in async_results]
    return results

# 3.d. concurrent.futures ProcessPoolExecutor
def executor_pool_map(numbers):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(square, numbers))
    return results

# Timing function to measure execution time
def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

if __name__ == '__main__':
    # Test with 10^6 numbers first
    count = 10**6
    numbers = create_numbers(count)
    chunk_size = 1000  # Experiment with different chunk sizes
    chunks = chunk_list(numbers, chunk_size)

    print(f"Testing with {count} numbers:")

    res, t = time_function(sequential_square, numbers)
    print(f"Sequential for loop: {t:.4f} seconds")

    # Warning: Creating one process per number is very heavy; try with a smaller subset.
    subset = numbers[:1000]  # For demonstration only.
    res, t = time_function(mp_individual_process, subset)
    print(f"Multiprocessing individual processes (1000 numbers): {t:.4f} seconds")

    res, t = time_function(mp_pool_map, numbers)
    print(f"Multiprocessing Pool map(): {t:.4f} seconds")

    res, t = time_function(mp_pool_apply, numbers)
    print(f"Multiprocessing Pool apply_async(): {t:.4f} seconds")

    res, t = time_function(executor_pool_map, numbers)
    print(f"concurrent.futures ProcessPoolExecutor: {t:.4f} seconds")

    # Now repeat with 10^7 numbers (you may need more resources; adjust if needed)
    count = 10**7
    numbers = create_numbers(count)
    print(f"\nTesting with {count} numbers:")

    res, t = time_function(sequential_square, numbers)
    print(f"Sequential for loop: {t:.4f} seconds")

    res, t = time_function(mp_pool_map, numbers)
    print(f"Multiprocessing Pool map(): {t:.4f} seconds")

    # For asynchronous testing with a pool:
    res, t = time_function(mp_pool_apply, numbers)
    print(f"Multiprocessing Pool apply_async(): {t:.4f} seconds")

    res, t = time_function(executor_pool_map, numbers)
    print(f"concurrent.futures ProcessPoolExecutor: {t:.4f} seconds")

    # Conclusions: 
    # - Creating one process per number (especially with 10^6 or 10^7 numbers) is not practical.
    # - The Pool-based methods (map, apply_async) and ProcessPoolExecutor are efficient ways to parallelize CPU-bound tasks.
    # - You should compare timings to see speedup and overhead, and note that for smaller tasks the overhead might overshadow benefits.


