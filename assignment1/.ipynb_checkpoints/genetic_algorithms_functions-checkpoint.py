import numpy as np
import random

def calculate_fitness(route, distance_matrix):
    """
    Calculates the total distance of a route.
    
    Parameters:
        route (list): The route to evaluate.
        distance_matrix (np.ndarray): The distance matrix.
    
    Returns:
        float: The negative total distance (to be minimized).
    """
    total_distance = 0
    for i in range(len(route) - 1):
        node1, node2 = route[i], route[i + 1]
        if distance_matrix[node1, node2] == 100000:
            return -1e6  # Penalty for infeasible route
        total_distance += distance_matrix[node1, node2]
    return -total_distance

def repair_route(route, distance_matrix, max_attempts=10):
    """
    Repairs a route by swapping nodes to eliminate infeasible legs.
    
    Parameters:
        route (list): The route to repair.
        distance_matrix (np.ndarray): The distance matrix.
        max_attempts (int): Maximum number of repair attempts.
    
    Returns:
        list: The repaired route.
    """
    repaired_route = route[:]
    for _ in range(max_attempts):
        feasible = True

        for i in range(len(repaired_route) - 1):
            if distance_matrix[repaired_route[i], repaired_route[i + 1]] == 100000:
                feasible = True


                # 🔴 Find the closest valid node to swap
                for j in range(i + 2, len(repaired_route)):
                    if (distance_matrix[repaired_route[i], repaired_route[j]] < 100000 and
                        distance_matrix[repaired_route[j], repaired_route[i + 1]] < 100000):
                        # ✅ Swap valid node
                        repaired_route[i + 1], repaired_route[j] = repaired_route[j], repaired_route[i + 1]
                        feasible = True
                        break
                
                if not feasible:
                    print(f"⚠ Unable to repair route segment: {repaired_route[i]} -> {repaired_route[i+1]}")
                    return route  # Return original if no feasible swap found
        
        if feasible:
            break
    
    return repaired_route

    #             swap_candidates = [j for j in range(i + 2, len(repaired_route))
    #                                if distance_matrix[repaired_route[i], repaired_route[j]] < 100000]
    #             if swap_candidates:
    #                 swap_idx = random.choice(swap_candidates)
    #                 repaired_route[i + 1], repaired_route[swap_idx] = repaired_route[swap_idx], repaired_route[i + 1]
    #             else:
    #                 return route  # Return original if repair fails
    # return repaired_route

def heuristic_route(distance_matrix):
    """
    Generates a route using a nearest-neighbor heuristic.
    
    Parameters:
        distance_matrix (np.ndarray): The distance matrix.
    
    Returns:
        list: A candidate route.
    """
    num_nodes = distance_matrix.shape[0]
    current, route = 0, [0]
    unvisited = set(range(1, num_nodes))

    while unvisited:
        # Get the closest neighbors, sorted by distance
        neighbors = sorted(
            [(node, distance_matrix[current, node]) for node in unvisited if distance_matrix[current, node] < 100000],
            key=lambda x: x[1]
        )
      
        if neighbors:
            # Pick randomly among the top 3 closest
            next_node = random.choice(neighbors[:min(3, len(neighbors))])[0]
        else:
            # If no valid neighbors, randomly pick from unvisited
            next_node = unvisited.pop()
        
        route.append(next_node)
        unvisited.discard(next_node)
        current = next_node


    route.append(0)  # Return to depot
    
    #print(f"🛠️ Generated Route: {route}")  # 🔍 Debugging line
    return route

def select_in_tournament(population, scores, num_tournaments=4, tournament_size=3):
    """
    Selects individuals using tournament selection.
    
    Parameters:
        population (list): The population of routes.
        scores (np.ndarray): The fitness scores.
        num_tournaments (int): Number of tournaments.
        tournament_size (int): Size of each tournament.
    
    Returns:
        list: Selected individuals.
    """
    selected = []
    for _ in range(num_tournaments):
        idx = np.random.choice(len(population), size=min(len(population), tournament_size), replace=False)
        best_idx = idx[np.argmax([scores[i] for i in idx])]
        selected.append(population[best_idx])
    return selected

def order_crossover(parent1, parent2):
    """
    Performs Order Crossover (OX1) while ensuring a valid child route.
    
    Parameters:
        parent1 (list): The first parent route.
        parent2 (list): The second parent route.
    
    Returns:
        list: A valid child route.
    """
    if len(parent1) != len(parent2):
        print(f"⚠ Parent length mismatch: {len(parent1)} vs {len(parent2)}. Returning parent1.")
        return parent1[:]  # Return parent1 as fallback

    size = len(parent1)
    child = [-1] * size

    start, end = sorted(random.sample(range(1, size - 1), 2))
    child[start:end] = parent1[start:end]

    remaining_nodes = [node for node in parent2 if node not in child]

    insert_idx = 0
    for i in range(size):
        if child[i] == -1:
            if insert_idx < len(remaining_nodes):  # ✅ Ensure insert index is within range
                child[i] = remaining_nodes[insert_idx]
                insert_idx += 1
            else:
                print("⚠ Insert index exceeded remaining nodes. Returning parent1.")
                return parent1[:]  # Fallback to parent1 in case of an issue

    return child


def mutate(route, mutation_rate=0.1):
    """
    Mutates a route by swapping two random nodes (excluding depot).
    
    Parameters:
        route (list): The route to mutate.
        mutation_rate (float): Probability of mutation.
    
    Returns:
        list: The mutated route.
    """
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(1, len(route) - 1), 2))
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def generate_unique_population(population_size, num_nodes, distance_matrix):
    """
    Generates a unique population of routes.
    
    Parameters:
        population_size (int): Size of the population.
        num_nodes (int): Number of nodes.
        distance_matrix (np.ndarray): The distance matrix.
    
    Returns:
        list: A list of unique routes.
    """
    population = []
    seen = set()
    attempts = 0
    max_attempts = population_size * 30  # 🔥 Increased attempt limit
    print('population seen, attempts', seen, attempts, population)

    while len(population) < population_size and attempts < max_attempts:
        route = heuristic_route(distance_matrix)
        route = repair_route(route, distance_matrix)
        route_tuple = tuple(route)
        
        if route_tuple not in seen:
            seen.add(route_tuple)
            population.append(route)
        else:
            print(f"⚠️ Duplicate route generated, skipping. Attempt {attempts}/{max_attempts}")
        attempts += 1
    if len(population) < population_size:
        print(f"⚠️ WARNING: Only {len(population)} routes generated out of {population_size}!")

    return population

def genetic_algorithm(distance_matrix, population_size=100, mutation_rate=0.2, max_generations=500):
    """
    Genetic Algorithm for route optimization.
    
    Parameters:
        distance_matrix (np.ndarray): The distance matrix.
        population_size (int): Number of individuals in population.
        mutation_rate (float): Probability of mutation.
        max_generations (int): Number of generations.
    
    Returns:
        tuple: Best route found and its fitness.
    """
    population = generate_unique_population(population_size, distance_matrix.shape[0], distance_matrix)
    best_route, best_fitness = None, float('-inf')

    for generation in range(max_generations):
        scores = np.array([calculate_fitness(route, distance_matrix) for route in population])
        parents = select_in_tournament(population, scores)

        next_generation = []
        while len(next_generation) < population_size:
            # p1, p2 = random.sample(parents, 2)
            print(f"🔍 Selected Parents: \nP1: {p1}\nP2: {p2}")
            if len(population) < 2:
                print("⚠ Not enough individuals for crossover. Skipping generation.")
                break
            else:
                p1, p2 = random.sample(population, 2)
            if len(p1) == len(p2):  # ✅ Ensure both parents have the same length
                child = mutate(order_crossover(p1, p2), mutation_rate)
            else:
                print("⚠ Mismatched parents in crossover, skipping this pair.")
                child = p1[:]  # Fallback to parent1
            print(f"🔍 Parent 1: {p1}")
            print(f"🔍 Parent 2: {p2}")

            child = mutate(order_crossover(p1, p2), mutation_rate)
            if tuple(child) not in next_generation:
                next_generation.append(child)

        population = next_generation
        best_in_gen_idx = np.argmax(scores)
        if scores[best_in_gen_idx] > best_fitness:
            best_route, best_fitness = population[best_in_gen_idx], scores[best_in_gen_idx]

        if generation % 50 == 0:
            print(f"Generation {generation}: Best Distance = {-best_fitness}")

    return best_route, -best_fitness






def mutate(route, mutation_rate=0.1):
    """
    Mutates a route by swapping two random nodes (excluding depot).
    
    Parameters:
        route (list): The route to mutate.
        mutation_rate (float): Probability of mutation.
    
    Returns:
        list: The mutated route.
    """
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(1, len(route) - 1), 2))
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def generate_unique_population(population_size, num_nodes, distance_matrix):
    """
    Generates a unique population of routes.
    
    Parameters:
        population_size (int): Size of the population.
        num_nodes (int): Number of nodes.
        distance_matrix (np.ndarray): The distance matrix.
    
    Returns:
        list: A list of unique routes.
    """
    population = []
    seen = set()
    attempts = 0
    
    while len(population) < population_size and attempts < population_size * 10:
        route = heuristic_route(distance_matrix)
        route = repair_route(route, distance_matrix)
        route_tuple = tuple(route)
        if route_tuple not in seen:
            seen.add(route_tuple)
            population.append(route)
        attempts += 1
    
    return population




# import numpy as np
# import random

# def calculate_fitness(route, distance_matrix):
#     """
#     Calculates the total distance of a route.
    
#     Parameters:
#         route (list): The route to evaluate.
#         distance_matrix (np.ndarray): The distance matrix.
    
#     Returns:
#         float: The negative total distance (to be minimized).
#     """
#     total_distance = 0
#     for i in range(len(route) - 1):
#         node1, node2 = route[i], route[i + 1]
#         if distance_matrix[node1, node2] == 100000:
#             return -1e6  # Penalty for infeasible route
#         total_distance += distance_matrix[node1, node2]
#     return -total_distance

# def repair_route(route, distance_matrix, max_attempts=10):
#     """
#     Repairs a route by swapping nodes to eliminate infeasible legs.
    
#     Parameters:
#         route (list): The route to repair.
#         distance_matrix (np.ndarray): The distance matrix.
#         max_attempts (int): Maximum number of repair attempts.
    
#     Returns:
#         list: The repaired route.
#     """
#     repaired_route = route.copy()
#     for _ in range(max_attempts):
#         feasible = True
#         for i in range(len(repaired_route) - 1):
#             if distance_matrix[repaired_route[i], repaired_route[i + 1]] == 100000:
#                 feasible = False
#                 # Try to find a swap candidate
#                 for j in range(i + 2, len(repaired_route)):
#                     if (distance_matrix[repaired_route[i], repaired_route[j]] < 100000 and
#                         distance_matrix[repaired_route[j - 1], repaired_route[i + 1]] < 100000):
#                         repaired_route[i + 1], repaired_route[j] = repaired_route[j], repaired_route[i + 1]
#                         break
#                 else:
#                     # No feasible swap found
#                     return route  # Return original route if repair fails
#         if feasible:
#             break
#     return repaired_route

# def heuristic_route(distance_matrix):
#     """
#     Generates a route using a nearest-neighbor heuristic.
    
#     Parameters:
#         distance_matrix (np.ndarray): The distance matrix.
    
#     Returns:
#         list: A candidate route.
#     """
#     num_nodes = distance_matrix.shape[0]
#     current = 0
#     route = [current]
#     unvisited = set(range(1, num_nodes))
    
#     while unvisited:
#         feasible_neighbors = [(node, distance_matrix[current, node]) 
#                              for node in unvisited if distance_matrix[current, node] < 100000]
#         if feasible_neighbors:
#             next_node = min(feasible_neighbors, key=lambda x: x[1])[0]
#         else:
#             next_node = unvisited.pop()
#         route.append(next_node)
#         unvisited.remove(next_node)
#         current = next_node
    
#     route.append(0)  # Return to depot
#     return route

# def select_in_tournament(population, scores, number_tournaments=4, tournament_size=3):
#     """
#     Selects individuals using tournament selection.
    
#     Parameters:
#         population (list): The population of routes.
#         scores (np.ndarray): The fitness scores.
#         number_tournaments (int): Number of tournaments to run.
#         tournament_size (int): Size of each tournament.
    
#     Returns:
#         list: Selected individuals.
#     """
#     selected = []
#     tournament_size=min(len(population),tournament_size)
#     for _ in range(number_tournaments):
#         idx = np.random.choice(len(population), size=tournament_size, replace=False)
#         best_idx = idx[np.argmax([scores[i] for i in idx])]
#         selected.append(population[best_idx])
#     return selected





# def order_crossover(parent1, parent2):
#     """
#     Performs order crossover between two parents.
    
#     Parameters:
#         parent1 (list): The first parent route.
#         parent2 (list): The second parent route.
    
#     Returns:
#         list: The offspring route.
#     """
#     size = len(parent1)
#     start, end = sorted(random.sample(range(1, size - 1), 2))  # Exclude depot
#     child = [0] + [-1] * (size - 2) + [0]  # Initialize with depots
#     child[start:end] = parent1[start:end]  # Copy segment from parent1
    
#     remaining_nodes = [node for node in parent2 if node not in child]
#     insert_pos = 1
#     for node in remaining_nodes:
#         while child[insert_pos] != -1:
#             insert_pos += 1
#         child[insert_pos] = node
    
#     return child

# import numpy as np
# import random

# def calculate_fitness(route,
#                       distance_matrix):
#     """
#     calculate_fitness function: total distance traveled by the car.

#     Parameters:
#         - route (list): A list representing the order of nodes visited in the route.
#         - distance_matrix (numpy.ndarray): A matrix of the distances between nodes.
#             A 2D numpy array where the element at position [i, j] represents the distance between node i and node j.
#     Returns:
#         - float: The negative total distance traveled (negative because we want to minimize distance).
#            Returns a large negative penalty if the route is infeasible.
#     """
#     total_distance = 0
    
# # Ensure that the route includes the depot at the start and end if needed.
#     for i in range(len(route) - 1):
#         node1 = route[i]
#         node2 = route[i+1]
#         distance = distance_matrix[node1][node2]
#         # If the distance indicates no connection, return a penalty.
#         if distance == 100000:  # or adjust to 10000 if that’s your marker
#             return -1e6
#         total_distance += distance
#     return -total_distance

# def repair_route(route, distance_matrix):
#     """
#     Attempts to repair a route by checking for infeasible legs (where distance == 100000)
#     and swapping nodes in hopes of replacing the infeasible connection.
    
#     Parameters:
#         route (list): A candidate route.
#         distance_matrix (np.ndarray): The distance matrix.
    
#     Returns:
#         list: The repaired route (or the original route if repair fails).
#     """
#     repaired_route = route.copy()
#     n = len(repaired_route)
#     for i in range(n - 1):
#         if distance_matrix[repaired_route[i], repaired_route[i+1]] == 100000:
#             # Try to find a swap candidate further along the route
#             for j in range(i+2, n):
#                 if (distance_matrix[repaired_route[i], repaired_route[j]] < 100000 and 
#                     distance_matrix[repaired_route[j-1], repaired_route[i+1]] < 100000):
#                     # Swap the nodes at positions i+1 and j
#                     repaired_route[i+1], repaired_route[j] = repaired_route[j], repaired_route[i+1]
#                     break  # Exit inner loop once a swap is made
#     return repaired_route

# def heuristic_route(distance_matrix):
#     """
#     Generates a route using a nearest-neighbor heuristic.
#     Starts at the depot (node 0), then repeatedly chooses the nearest unvisited node
#     that is connected (distance < 100000). If no feasible neighbor is found,
#     a random unvisited node is selected.
#     Finally, the depot is appended at the end.
    
#     Parameters:
#         distance_matrix (np.ndarray): The distance matrix.
    
#     Returns:
#         list: A candidate route.
#     """
#     num_nodes = distance_matrix.shape[0]
#     current = 0
#     route = [current]
#     unvisited = set(range(1, num_nodes))
    
#     while unvisited:
#         # Get all connected (feasible) neighbors among unvisited nodes
#         feasible_neighbors = [(node, distance_matrix[current, node]) 
#                               for node in unvisited if distance_matrix[current, node] < 100000]
#         if feasible_neighbors:
#             # Choose the neighbor with the minimum distance
#             next_node = min(feasible_neighbors, key=lambda x: x[1])[0]
#         else:
#             # If no feasible neighbor, pick a random one (route may become infeasible)
#             next_node = unvisited.pop()
#             route.append(next_node)
#             current = next_node
#             continue
#         route.append(next_node)
#         unvisited.remove(next_node)
#         current = next_node
    
#     # Append depot at the end
#     route.append(0)
#     return route


# def select_in_tournament(population,
#                          scores,
#                          number_tournaments=4,
#                          tournament_size=3):
#     """
#     Tournament selection for genetic algorithm.

#     Parameters:
#         - population (list): The current population of routes.
#         - scores (np.array): The calculate_fitness scores corresponding to each individual in the population.
#         - number_tournaments (int): The number of the tournamnents to run in the population.
#         - tournament_size (int): The number of individual to compete in the tournaments.

#     Returns:
#         - list: A list of selected individuals for crossover.
#     """
#     selected = []
    
#     n_individuals = len(population)
#     for _ in range(number_tournaments):
#         # Randomly select tournament_size individuals (without replacement)
#         idx = np.random.choice(n_individuals, size=tournament_size, replace=False)
#         # Among the selected, find the index with the maximum fitness value.
#         # (Remember: since fitness is negative total distance, a higher value means a lower total distance.)
#         best_idx = idx[np.argmax([scores[i] for i in idx])]
#         selected.append(population[best_idx])
#     return selected    


# # def order_crossover(parent1, parent2):
# #     """
# #     Order crossover (OX) for permutations.

# #     Parameters:
# #         - parent1 (list): The first parent route.
# #         - parent2 (list): The second parent route.

# #     Returns:
# #         - list: The offspring route generated by the crossover.
# #     """
# #     size = len(parent1)
# #     start, end = sorted(np.random.choice(range(size), 2, replace=False))
# #     offspring = [None] * size
# #     offspring[start:end + 1] = parent1[start:end + 1]
# #     fill_values = [x for x in parent2 if x not in offspring[start:end + 1]]
# #     idx = 0
# #     for i in range(size):
# #         if offspring[i] is None:
# #             offspring[i] = fill_values[idx]
# #             idx += 1
# #     return offspring

# def order_crossover(parent1, parent2):
#     """
#     Performs Order Crossover (OX1) between two parents.
#     Ensures that offspring inherits a contiguous section from one parent and
#     fills the remaining slots from the other parent while preserving order.
    
#     Parameters:
#         parent1 (list): The first parent route.
#         parent2 (list): The second parent route.
    
#     Returns:
#         list: The generated offspring route.
#     """
#     size = len(parent1)
#     start, end = sorted(random.sample(range(1, size - 1), 2))  # Ensure depot (0) is fixed

#     child = [-1] * size
#     child[0], child[-1] = 0, 0  # Depot remains fixed
#     child[start:end] = parent1[start:end]  # Copy a segment from parent1

#     remaining_nodes = [node for node in parent2 if node not in child]  # Nodes not yet in child

#     insert_pos = 1
#     for node in remaining_nodes:
#         while child[insert_pos] != -1:
#             insert_pos += 1
#         child[insert_pos] = node

#     return child


# # def mutate(route,
# #            mutation_rate = 0.1):
# #     """
# #     Mutation operator: swap two nodes in the route.

# #     Parameters:
# #         - route (list): The route to mutate.
# #         - mutation_rate (float): The chance to mutate an individual.
# #     Returns:
# #         - list: The mutated route.
# #     """
# #     if np.random.rand() < mutation_rate:
# #         i, j = np.random.choice(len(route), 2, replace=False)
# #         route[i], route[j] = route[j], route[i]
# #     return route

# def mutate(route, mutation_rate=0.01):
#     """
#     Mutates a route by swapping two random nodes (excluding the depot).
    
#     Parameters:
#         route (list): The route to mutate.
#         mutation_rate (float): The probability of mutation.
    
#     Returns:
#         list: The mutated route.
#     """
#     if random.random() < mutation_rate:
#         idx1, idx2 = sorted(random.sample(range(1, len(route) - 1), 2))  # Exclude depot (0)
#         route[idx1], route[idx2] = route[idx2], route[idx1]
#     return route


# def generate_unique_population(population_size, num_nodes, distance_matrix):
#     """
#     Generate a unique population of individuals for a genetic algorithm.

#     Each individual in the population represents a route in a graph, where the first node is fixed (0) and the 
#     remaining nodes are a permutation of the other nodes in the graph. This function ensures that all individuals
#     in the population are unique.

#     Parameters:
#         - population_size (int): The desired size of the population.
#         - num_nodes (int): The number of nodes in the graph, including the starting node.

#     Returns:
#         - list of lists: A list of unique individuals, where each individual is represented as a list of node indices.
#     """
#     population = []
#     seen=set()
#     attempts = 0

#     while len(population) < population_size and attempts < population_size * 10:
#         route = heuristic_route(distance_matrix)
#         route = repair_route(route, distance_matrix)
#         # Check uniqueness by converting the route list to a tuple.
#         route_tuple = tuple(route)
#         if route_tuple not in seen:
#             seen.add(route_tuple)
#             population.append(route)
#         attempts += 1
#     return population


#     # for _ in range(population_size):
#     #     # Generate a permutation of nodes 1 through num_nodes-1.
#     #     middle_route = list(np.random.permutation(range(1, num_nodes)))
#     #     # Add the depot (node 0) at the beginning and end.
#     #     route = [0] + middle_route + [0]
#     #     population.append(route)
#     # return population
    
#     # population = set()
#     # while len(population) < population_size:
#     #     individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
#     #     population.add(tuple(individual))
#     # return [list(ind) for ind in population]
