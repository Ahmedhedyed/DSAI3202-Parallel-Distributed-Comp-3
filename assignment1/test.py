from concurrent.futures import ProcessPoolExecutor

def square(n):
    return n * n

if __name__ == '__main__':
    numbers = range(10)  # Small set for testing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(square, numbers))
    print(results)
