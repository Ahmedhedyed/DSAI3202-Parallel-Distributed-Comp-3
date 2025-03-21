from mpi4py import MPI
import numpy as np
import time
import genetic_algorithms_functions as gaf  # Import your functions
import pandas as pd

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#df = pd.read_csv("city_distances_updated.csv", header=None)
df = pd.read_csv("city_distances_extended.csv", header=None)

#print(df.describe())  # Check for large values
print(df.shape)
print(df.isnull().sum().sum())  # Check for NaNs
print(df.max().max())  # Find the largest value

def parallel_fitness_evaluation(local_population, distance_matrix):
    """
    Evaluates the fitness of a population chunk in parallel using MPI.
    Each process computes the fitness for its assigned portion.

    Parameters:
        local_population (list): The portion of the population assigned to this rank.
        distance_matrix (np.ndarray): The distance matrix.

    Returns:
        list: Fitness values for the entire population (only at root process).
    """
    print(f"🔍 Rank {rank}: Processing {len(local_population)} individuals")

    # Compute local fitness values
    local_fitness = [gaf.calculate_fitness(route, distance_matrix) for route in local_population]

    # Gather all local fitness results at the root process
    all_fitness = comm.gather(local_fitness, root=0)

    if rank == 0:
        fitness = [f for sublist in all_fitness for f in sublist]  # Flatten list
        return fitness
    return None  # Other ranks return None

if __name__ == '__main__':
    print(f"📌 Rank {rank} started execution")

    if rank == 0:
          
        # Load distance matrix and generate the initial population
        #distance_matrix = np.loadtxt("city_distances_updated.csv", delimiter=",")
        distance_matrix = np.loadtxt("city_distances_extended.csv", delimiter=",")
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            print(f"⚠️ Fixing non-square matrix {distance_matrix.shape}...")
            min_dim = min(distance_matrix.shape)
            distance_matrix = distance_matrix[:min_dim, :min_dim]  # Make it square
        print("Matrix Shape ", distance_matrix.shape)

        num_nodes = distance_matrix.shape[0]
        print(f"📌 Distance Matrix Shape: {distance_matrix.shape}, Nodes: {num_nodes}")
    
        # 🔴 FIX: Ensure the population is at least `size * 10`
        population_size = max(500, size * 10)  # Make sure there are at least `size * 10` individuals
        print(f"🔍 Rank 0: Attempting to generate {population_size} individuals...")

        # ✅ Ensure enough unique routes are generated
        population = gaf.generate_unique_population(population_size, num_nodes, distance_matrix)
        print(f"🔍 Rank 0: Generated Population Size = {len(population)}")
        # Split population into chunks for each process
        chunk_size = (len(population) + size - 1) // size  
        chunks = [population[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
        print(f"✅ Population split into {size} chunks")

        # 🔴 Fix: Distribute any remaining individuals
        remainder = len(population) % size
        for i in range(remainder):
            chunks[i].append(population[-(i + 1)])
        
    else:
        distance_matrix = None
        chunks = None

    start_time = comm.bcast(time.time() if rank == 0 else None, root=0)  # Broadcast start time

    # Broadcast the distance matrix to all processes
    distance_matrix = comm.bcast(distance_matrix, root=0)

    # Scatter the population chunks across ranks
    local_population = comm.scatter(chunks, root=0)
    print(f"🔍 Rank {rank}: Received population size = {len(local_population)}")

    # Perform parallel fitness evaluation
    fitness = parallel_fitness_evaluation(local_population, distance_matrix)

    if rank == 0:
        end_time = time.time()
        print(f"✅ Parallel fitness evaluation completed in {end_time - start_time:.4f} seconds")
        print("🔍 Sample Fitness Values:", fitness[:10])  # Print only the first 10 fitness values

        # ✅ Select the best individual
        best_index = np.argmax(fitness)
        best_route = population[best_index]
        best_fitness = fitness[best_index]
        print(f"🏁 Best Distance: {-best_fitness}")


        # ✅ Save the best result to file
        with open("best_route_100.txt", "w") as f:
        #with open("best_route_32.txt", "w") as f:
            f.write(f"Best Route: {best_route}\n")
            f.write(f"Best Distance: {-best_fitness}\n")
            

