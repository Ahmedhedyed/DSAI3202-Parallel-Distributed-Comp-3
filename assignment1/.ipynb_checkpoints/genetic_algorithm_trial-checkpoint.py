import numpy as np
import pandas as pd
from genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population
)

# Load the distance matrix
distance_matrix = pd.read_csv('city_distances.csv').to_numpy()

# Parameters
num_nodes = distance_matrix.shape[0]
population_size = 1000  # Reduce for efficiency
mutation_rate = 0.1
num_generations = 200
stagnation_limit = 5  # Max generations without improvement
adaptive_mutation_increase = 0.05  # Increase mutation rate if stuck
infeasible_penalty = 1e6  # Penalize infeasible solutions

# Generate initial population
np.random.seed(42)
population = generate_unique_population(population_size, num_nodes, distance_matrix)

# Track best fitness
best_fitness = float('inf')
stagnation_counter = 0

# Function to validate parents before crossover
def is_valid_parent(parent):
    """ Check if parent has enough nodes for crossover """
    return len(parent) > 3  # Must have at least start, 1 city, and end

# Function to repair broken routes
def repair_route(route, num_nodes):
    """ Fix routes that lost cities during mutation/crossover """
    city_set = set(range(1, num_nodes))  # All cities except depot
    missing_cities = list(city_set - set(route))
    
    # Replace duplicates with missing cities
    seen = set()
    for i in range(1, len(route) - 1):
        if route[i] in seen:
            if missing_cities:
                route[i] = missing_cities.pop(0)  # Replace with a missing city
        seen.add(route[i])
    
    return route

# Main GA loop
for generation in range(num_generations):
    # Evaluate fitness
    fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])

    # Track best fitness
    current_best_fitness = np.min(fitness_values)
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    # Adaptive mutation on stagnation
    if stagnation_counter >= stagnation_limit:
        print(f"⚠ Stagnation detected at generation {generation}. Increasing mutation rate.")
        mutation_rate += adaptive_mutation_increase
        stagnation_counter = 0

    # Tournament selection
    tournament_size = min(3, len(population))  # Ensure valid selection
    selected = select_in_tournament(population, fitness_values, number_tournaments=100, tournament_size=tournament_size)

    # Crossover with validity check
    offspring = []
    for i in range(0, len(selected) - 1, 2):
        parent1, parent2 = selected[i], selected[i + 1]

        # Ensure valid parents
        if not is_valid_parent(parent1) or not is_valid_parent(parent2):
            print(f"⚠ Skipping crossover due to short parents: {parent1}, {parent2}")
            offspring.append(parent1[:])  # Keep the original
            continue

        try:
            # Extract inner nodes for crossover
            inner_parent1, inner_parent2 = parent1[1:-1], parent2[1:-1]

            # Ensure parents have enough nodes
            if len(inner_parent1) >= 3 and len(inner_parent2) >= 3:
                route1 = [0] + order_crossover(inner_parent1, inner_parent2) + [0]
                offspring.append(repair_route(route1, num_nodes))  # Repair if needed
            else:
                print("⚠ Parents too short for crossover. Skipping.")
                offspring.append(parent1[:])
        except Exception as e:
            print(f"⚠ Error during crossover: {e}")
            offspring.append(parent1[:])

    # Apply mutation with repairs
    mutated_offspring = [repair_route(mutate(route, mutation_rate), num_nodes) for route in offspring]

    # Replacement: Only replace worst individuals if offspring is better
    num_replacements = min(len(mutated_offspring), len(population))
    for i, idx in enumerate(np.argsort(fitness_values)[::-1][:num_replacements]):
        if calculate_fitness(mutated_offspring[i], distance_matrix) < fitness_values[idx]:
            population[idx] = mutated_offspring[i]

    # Ensure population uniqueness
    new_population = set(map(tuple, population))
    while len(new_population) < population_size:
        new_individual = [0] + list(np.random.permutation(np.arange(1, num_nodes))) + [0]
        new_population.add(tuple(new_individual))
    population = [list(ind) for ind in new_population]

    # Print best fitness
    print(f"Generation {generation}: Best Fitness = {current_best_fitness}")

# Final evaluation
fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])

# Output best solution
best_idx = np.argmin(fitness_values)
best_solution = population[best_idx]
print("Best Solution:", best_solution)
print("Total Distance:", calculate_fitness(best_solution, distance_matrix))

# import numpy as np
# import pandas as pd
# from genetic_algorithms_functions import calculate_fitness, \
#     select_in_tournament, order_crossover, mutate, \
#     generate_unique_population

# # Load the distance matrix
# distance_matrix = pd.read_csv('city_distances.csv').to_numpy()

# # Parameters
# num_nodes = distance_matrix.shape[0]
# population_size = 10000
# num_tournaments = 4  # Number of tournaments to run
# mutation_rate = 0.1
# num_generations = 200
# infeasible_penalty = 1e6  # Penalty for infeasible routes
# stagnation_limit = 5  # Number of generations without improvement before regeneration

# # Generate initial population: each individual is a route starting at node 0
# np.random.seed(42)  # For reproducibility
# population = generate_unique_population(population_size, num_nodes, distance_matrix)

# # Initialize variables for tracking stagnation
# best_calculate_fitness = int(1e6)
# stagnation_counter = 0

# # Main GA loop
# for generation in range(num_generations):
#     # Evaluate calculate_fitness
#     calculate_fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])

#     # Check for stagnation
#     current_best_calculate_fitness = np.min(calculate_fitness_values)
#     if current_best_calculate_fitness < best_calculate_fitness:
#         best_calculate_fitness = current_best_calculate_fitness
#         stagnation_counter = 0
#     else:
#         stagnation_counter += 1

#     # Regenerate population if stagnation limit is reached, keeping the best individual
#     if stagnation_counter >= stagnation_limit:
#         print(f"Regenerating population at generation {generation} due to stagnation")
#         best_individual = population[np.argmin(calculate_fitness_values)]
#         population = generate_unique_population(population_size - 1, num_nodes, distance_matrix)
#         population.append(best_individual)
#         stagnation_counter = 0
#         continue  # Skip the rest of the loop for this generation

#     # Selection, crossover, and mutation
#     selected = select_in_tournament(population, calculate_fitness_values, number_tournaments=100)

#     offspring = []
#     for i in range(0, len(selected) - 1, 2):  # Ensure an even number of parents
#         parent1, parent2 = selected[i], selected[i + 1]
        
#         # Ensure crossover only operates on inner nodes (excluding depots)
#         if len(parent1) > 3 and len(parent2) > 3:  
#             route1 = order_crossover(parent1[1:-1], parent2[1:-1])
#             offspring.append([0] + route1 + [0])  # Ensure depot is preserved
#         else:
#             offspring.append(parent1[:])  # Keep parent if crossover fails

#     # Apply mutation to offspring
#     mutated_offspring = [mutate(route, mutation_rate) for route in offspring]

#     # Replacement: Replace worst-performing individuals
#     num_replacements = min(len(mutated_offspring), len(population))
#     for i, idx in enumerate(np.argsort(calculate_fitness_values)[::-1][:num_replacements]):
#         population[idx] = mutated_offspring[i]

#     # Ensure population uniqueness
#     new_population = set(map(tuple, population))
#     while len(new_population) < population_size:
#         new_individual = [0] + list(np.random.permutation(np.arange(1, num_nodes))) + [0]
#         new_population.add(tuple(new_individual))
#     population = [list(ind) for ind in new_population]

#     # Print best calculate_fitness
#     print(f"Generation {generation}: Best calculate_fitness = {current_best_calculate_fitness}")

# # Update calculate_fitness_values for the final population
# calculate_fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])

# # Output the best solution
# best_idx = np.argmin(calculate_fitness_values)
# best_solution = population[best_idx]
# print("Best Solution:", best_solution)
# print("Total Distance:", calculate_fitness(best_solution, distance_matrix))






# # import numpy as np
# # import pandas as pd
# # from genetic_algorithms_functions import calculate_fitness, \
# #     select_in_tournament, order_crossover, mutate, \
# #     generate_unique_population
# # from genetic_algorithms_functions import heuristic_route, repair_route
# # import random

# # # Load the distance matrix
# # distance_matrix = pd.read_csv('city_distances_updated.csv').to_numpy()

# # # Parameters
# # num_nodes = distance_matrix.shape[0]
# # population_size = 10000
# # num_tournaments = 4  # Number of tournaments to run
# # mutation_rate = 0.1
# # num_generations = 200
# # infeasible_penalty = 1e6  # Penalty for infeasible routes
# # stagnation_limit = 5  # Number of generations without improvement before regeneration

# # # Generate initial population: each individual is a route starting at node 0
# # np.random.seed(42)  # For reproducibility
# # population = generate_unique_population(population_size, num_nodes, distance_matrix)

# # # Initialize variables for tracking stagnation
# # best_calculate_fitness = float('inf')  # Start with a very high negative value
# # # best_calculate_fitness = int(1e6)
# # stagnation_counter = 0

# # # Main GA loop
# # for generation in range(num_generations):
# #     # Evaluate calculate_fitness
# #     calculate_fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])

# #     # Check for stagnation
# #     current_best_calculate_fitness = np.min(calculate_fitness_values)
# #     if current_best_calculate_fitness < best_calculate_fitness:
# #         best_calculate_fitness = current_best_calculate_fitness
# #         stagnation_counter = 0
# #     else:
# #         stagnation_counter += 1

# #     # Regenerate population if stagnation limit is reached, keeping the best individual
# #     if stagnation_counter >= stagnation_limit:
# #         print(f"Regenerating population at generation {generation} due to stagnation")
# #         best_individual = population[np.argmin(calculate_fitness_values)]
# #         population = generate_unique_population(population_size - 1, num_nodes, distance_matrix)
# #         population.append(best_individual)
# #         stagnation_counter = 0
# #         continue  # Skip the rest of the loop for this generation

# #      # Ensure the tournament size is valid before calling select_in_tournament
# #     valid_tournament_size = min(3, len(population))  # Tournament size should never be larger than population

    
# #     if valid_tournament_size > 0:  # Ensure we don't pass 0 as tournament_size
# #         selected = select_in_tournament(population, calculate_fitness_values, number_tournaments=100, tournament_size=valid_tournament_size)
# #     else:
# #         selected = []  # If population is too small, selection returns an empty list


   

# #     # # valid_tournament_size = max(1, min(num_tournaments, len(population)))  # Ensure at least one individual is selected
    
# #     # selected = select_in_tournament(population, calculate_fitness_values, number_tournaments=valid_tournament_size)

# #     # Selection, crossover, and mutation
# #     # selected = select_in_tournament(population,
# #     #                                 calculate_fitness_values)
# #     offspring = []
# #     # for i in range(0, len(selected), 2):
# #     #     parent1, parent2 = selected[i], selected[i + 1]
# #     #     route1 = order_crossover(parent1[1:], parent2[1:])
# #     #     offspring.append([0] + route1)
# #     # mutated_offspring = [mutate(route, mutation_rate) for route in offspring]
# #     for i in range(0, len(selected) - 1, 2):  # Ensure an even number of parents
# #         parent1, parent2 = selected[i], selected[i + 1]
    
# #         # Ensure parents have enough nodes before crossover
# #         if len(parent1) > 4 and len(parent2) > 4:  
# #             # Ensure the selected portion is not too short for crossover
# #             if len(parent1[1:-1]) > 2 and len(parent2[1:-1]) > 2:
# #                 try:
# #                     route1 = [0] + order_crossover(parent1[1:-1], parent2[1:-1]) + [0]  # Preserve start & end depot
# #                     offspring.append(route1)
# #                 except IndexError:
# #                     print(f"⚠ Skipping invalid crossover between {parent1} and {parent2}")
# #                     offspring.append(parent1[:])  # Fallback to parent if crossover fails
# #             else:
# #                 # If the extracted portion is too short, avoid crossover and keep one parent
# #                 offspring.append(parent1[:])  
# #         else:
# #             # If parents are too short, just keep one of them as offspring
# #             offspring.append(parent1[:])  

        
# #     # for i in range(0, len(selected) - 1, 2):  # Ensure an even number of parents
# #     #     parent1, parent2 = selected[i], selected[i + 1]
# #     #     route1 = [0] + order_crossover(parent1[1:-1], parent2[1:-1]) + [0]  # Preserve start & end depot
# #     #     offspring.append(route1)
# #     # mutated_offspring = [mutate(route, mutation_rate) for route in offspring]  # Apply mutation
    
# #     mutated_offspring = [mutate(route, mutation_rate) for route in offspring]  # Apply mutation
# #     # Replacement: Replace the individuals that lost in the tournaments with the new offspring
# #     num_replacements = min(len(mutated_offspring), len(population))  # Prevent out-of-range errors
# #     # Replacement: Replace the individuals that lost in the tournaments with the new offspring
    
# #     for i, idx in enumerate(np.argsort(calculate_fitness_values)[::-1][:num_replacements]):
# #         population[idx] = mutated_offspring[i]

# #     # for i, idx in enumerate(np.argsort(calculate_fitness_values)[::-1][:len(mutated_offspring)]):
# #     #     population[idx] = mutated_offspring[i]

# #     # # Ensure population uniqueness
# #     # unique_population = set(tuple(ind) for ind in population)
# #     # while len(unique_population) < population_size:
# #     #     individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
# #     #     unique_population.add(tuple(individual))
# #     # population = [list(individual) for individual in unique_population]
# #     unique_population = set(tuple(ind) for ind in population)
# #     while len(unique_population) < population_size:
# #         individual = heuristic_route(distance_matrix)  # Generate heuristic-based route
# #         individual = repair_route(individual, distance_matrix)  # Ensure feasibility
# #         unique_population.add(tuple(individual))
# #     population = [list(individual) for individual in unique_population]

# #     # Print best calculate_fitness
# #     print(f"Generation {generation}: Best calculate_fitness = {current_best_calculate_fitness}")

# # # Update calculate_fitness_values for the final population
# # calculate_fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])

# # # Output the best solution
# # best_idx = np.argmin(calculate_fitness_values)
# # best_solution = population[best_idx]
# # print("Best Solution:", best_solution)
# # print("Total Distance:", calculate_fitness(best_solution, distance_matrix))
