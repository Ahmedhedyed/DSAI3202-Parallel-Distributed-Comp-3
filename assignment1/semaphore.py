import time
import random
import multiprocessing as mp

class ConnectionPool:
    def __init__(self, max_connections):
        # Create a list to simulate available connections (for example, connection IDs)
        self.connections = list(range(max_connections))
        # A semaphore initialized to the maximum number of connections available
        self.semaphore = mp.Semaphore(max_connections)
        # Lock to synchronize access to the connections list (optional for clarity)
        self.lock = mp.Lock()

    def get_connection(self):
        # Wait for a connection to be available (decrement semaphore)
        self.semaphore.acquire()
        with self.lock:
            # Remove and return a connection from the pool
            connection = self.connections.pop(0)
        return connection

    def release_connection(self, connection):
        with self.lock:
            # Add the connection back to the pool
            self.connections.append(connection)
        # Release the semaphore (increment count)
        self.semaphore.release()

def access_database(pool, process_id):
    print(f"Process {process_id} is waiting for a connection...")
    connection = pool.get_connection()
    print(f"Process {process_id} acquired connection {connection}")
    # Simulate database operation by sleeping a random time
    time.sleep(random.uniform(0.5, 2.0))
    print(f"Process {process_id} releasing connection {connection}")
    pool.release_connection(connection)

if __name__ == '__main__':
    # Set the maximum number of connections available in the pool
    max_connections = 3
    pool = ConnectionPool(max_connections)

    # Create a list of processes (more than max_connections to observe blocking)
    processes = []
    num_processes = 10
    for i in range(num_processes):
        p = mp.Process(target=access_database, args=(pool, i))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Observations to Discuss:
    # - If more processes try to access the pool than there are available connections,
    #   they will block on semaphore.acquire() until a connection is released.
    # - The semaphore prevents race conditions by ensuring that only max_connections number
    #   of processes can acquire a connection concurrently.
    # - The lock is used to synchronize access to the connections list when removing or adding connections.
    print("All processes have completed their database operations.")
