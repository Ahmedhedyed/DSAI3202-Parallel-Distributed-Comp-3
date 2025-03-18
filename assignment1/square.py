import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np 
from multiprocessing import cpu_count





# 1. Define a function to compute the square of a number.
def square(n):
    return n * n

def worker(idx, n, results):
    """Worker function defined at the top-level so it can be pickled."""
    results[idx] = square(n)




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
    num_worker=min(cpu_count(), len(numbers))
    processes = []
    chunk_size=len(numbers)//num_worker 
    manager = mp.Manager()
    results = manager.list([None] * len(numbers))  # Shared list to store results


    for i in range(num_worker):
        start=i*chunk_size
        end=start+chunk_size if i< num_worker-1 else len(numbers)
        p = mp.Process(target=sequential_square, args=(numbers[start:end],))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    return list(results)
    

# 3.c. Multiprocessing Pool using map() and apply()
def mp_pool_map(numbers):
    with mp.Pool() as pool:
        results = pool.map(square, numbers)
    return results

def mp_pool_apply(numbers):
    results = [None] * len(numbers)  # preallocate results

    # Here we simulate apply by calling apply_async in a loop and then collecting the results.
    def store_result(result, idx):
        results[idx] = result

    with mp.Pool() as pool:
        for i, n in enumerate(numbers):
            pool.apply_async(square, args=(n,), callback=lambda res, idx=i: store_result(res, idx))
        pool.close()
        pool.join()
        return results

def executor_pool_map(numbers, max_workers=None, chunk_size=None):
    max_workers = max_workers or cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(square, numbers, chunksize=chunk_size))
    return results

# # 3.d. concurrent.futures ProcessPoolExecutor
# def executor_pool_map(numbers):
#     with ProcessPoolExecutor() as executor:
#         results = list(executor.map(square, numbers))
#     return results

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
    # chunk_size = 1000  # Experiment with different chunk sizes
    # chunks = chunk_list(numbers, chunk_size)

    print(f"Testing with {count} numbers:")

    res, t = time_function(sequential_square, numbers)
    print(f"Sequential for loop: {t:.4f} seconds")

    # Warning: Creating one process per number is very heavy; try with a smaller subset.
    strat=time.time()
    res, t = time_function(mp_individual_process, numbers)
    print(f"Multiprocessing individual processes  {t:.4f} seconds")

    res, t = time_function(mp_pool_map, numbers)
    print(f"Multiprocessing Pool map(): {t:.4f} seconds")

    res, t = time_function(mp_pool_apply, numbers)
    print(f"Multiprocessing Pool apply_async(): {t:.4f} seconds")

    res, t = time_function(executor_pool_map, numbers, max_workers=4, chunk_size=1000)
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

    res, t = time_function(executor_pool_map, numbers, max_workers=4, chunk_size=1000)
    print(f"concurrent.futures ProcessPoolExecutor: {t:.4f} seconds")

# Conclusions: 
# - Creating one process per number (especially with 10^6 or 10^7 numbers) is not practical.
# - for indvidual proecesses we divded the numbers into chunkcs an created processe with help from cpu_count and assign each chunk to one proercss.
# - The Pool-based methods (map, apply_async) and ProcessPoolExecutor are efficient ways to parallelize CPU-bound tasks.
# - For smaller tasks the overhead might overshadow benefits.
# - Async apply pool took 76.409 second in the first run with 10^6 and 811.9733 seconds in the second run with 10^7 which is expected 
# in async proecsses as the will go sequanantioal insetad of parallel.

# Testing with 1000000 numbers:
# Sequential for loop: 0.0715 seconds
# Multiprocessing individual processes  26.2376 seconds
# Multiprocessing Pool map(): 1.3631 seconds
# Multiprocessing Pool apply_async(): 76.5310 seconds
# concurrent.futures ProcessPoolExecutor: 1.1699 seconds

# Testing with 10000000 numbers:
# Sequential for loop: 0.6980 seconds
# Multiprocessing Pool map(): 10.3894 seconds
# Multiprocessing Pool apply_async(): 811.9733 seconds
# concurrent.futures ProcessPoolExecutor: 11.2477 seconds


