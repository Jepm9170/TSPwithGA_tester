import random
import matplotlib.pyplot as plt

# Number of cities
num_cities = int(input("Enter the number of cities: "))
mutation_rate = float(input("Enter the mutation rate: "))

population_size = int(input("Enter the initial population size: "))

# Generate random distances between cities
distances = {}
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        city1 = chr(ord('A') + i)
        city2 = chr(ord('A') + j)
        distance = random.randint(1, 10000)  # Replace the range as per your requirements
        distances[(city1, city2)] = distance
        distances[(city2, city1)] = distance  # Add the reverse pair as well

# Function to get distance between two cities
def get_distance(city1, city2):
    if (city1, city2) in distances:
        return distances[(city1, city2)]
    elif (city2, city1) in distances:
        return distances[(city2, city1)]
    else:
        return random.randint(1, 10000)
'''
print("Generated distances:")
for cities, distance in distances.items():
    print(cities, "->", distance)
    # Test get_distance function
print("Distance between A and B:", get_distance('A', 'B'))
'''



# Create the initial population
def create_individual(cities):
    individual = cities.copy()
    random.shuffle(individual)
    return individual

# Calculate fitness score for an individual
def calculate_fitness(individual):
    score = 0
    for i in range(num_cities - 1):
        city1 = individual[i]
        city2 = individual[i + 1]
        distance = get_distance(city1, city2)
        score += distance
        #print(f"Distance between {city1} and {city2}: {distance}")

    #print(f"Total distance for individual {individual}: {score}")
    return score


def calculate_fitness_scores(population):
    fitness_scores = []
    total_fitness = 0

    for individual in population:
        score = calculate_fitness(individual)
        fitness_scores.append(score)
        total_fitness += score

    if total_fitness == 0:
        probabilities = [1 / len(population)] * len(population)
    else:
        probabilities = [score / total_fitness for score in fitness_scores]

    return fitness_scores, probabilities

# Perform selection using tournament selection
tournament_size=2
def tournament_selection(population, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)  # Randomly select a subset
        winner = min(tournament, key=calculate_fitness)  # Select the winner based on fitness
        selected.append(winner)
    return selected

# Perform crossover using uniform crossover
def crossover(parent1, parent2):
    child1 = []
    child2 = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2

# Perform mutation
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

# Get user input for the number of generations

num_generations = int(input("Enter the number of generations: "))

# Generate the initial population


# Initialize plot
plt.figure()

# Plot the cities
x = [random.uniform(0, 10) for _ in range(num_cities)]
y = [random.uniform(0, 10) for _ in range(num_cities)]
plt.scatter(x, y, color='red')

population = []
for _ in range(population_size):
    individual = random.sample(range(num_cities), num_cities)  # Generate a random individual
    population.append(individual)

best_distances = []
# Genetic algorithm main loop
for generation in range(num_generations):
    # Perform selection, crossover, and mutation to create the next generation
    fitness_scores, _ = calculate_fitness_scores(population)

    # Calculate and store the best distance for this generation
    best_distance = min(fitness_scores)
    best_distances.append(best_distance)

    probabilities = []
    total_fitness = sum(fitness_scores)

    if total_fitness == 0:
        # Set equal probabilities for all individuals if total_fitness is zero
        probabilities = [1 / len(population)] * len(population)
    else:
        probabilities = [score / total_fitness for score in fitness_scores]

     # Find the best individual from the current population
    best_individual = population[fitness_scores.index(min(fitness_scores))]

     # Append the best individual to the next generation
    next_generation = [best_individual]

    parents = tournament_selection(population, tournament_size)

    parent1 = None
    parent2 = None

    if len(parents) >= 2:
        parent1, parent2 = parents[:2]
    elif len(parents) == 1:
        parent1 = parents[0]
        parent2 = parents[0]  # You can choose to use the same parent for crossover




    while len(next_generation) < population_size:
        parents = tournament_selection(population, tournament_size)

        parent1 = None
        parent2 = None

        if len(parents) >= 2:
            parent1, parent2 = parents[:2]
        elif len(parents) == 1:
            parent1 = parents[0]
            parent2 = parents[0]  # You can choose to use the same parent for crossover

        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([mutation(child1, mutation_rate), mutation(child2, mutation_rate)])
    population = next_generation

    # Get the best individual from the current population
    best_individual = min(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_individual)
    print(f"Generation {generation + 1}: Best Distance = {best_fitness}")
    # Plot the best solution for each generation
    best_x = [x[i] for i in best_individual]
    best_y = [y[i] for i in best_individual]
    plt.plot(best_x, best_y, alpha=0.5)



# Plot the best solution from the final population
best_individual = min(population, key=calculate_fitness)
best_x = [x[i] for i in best_individual]
best_y = [y[i] for i in best_individual]
plt.plot(best_x, best_y, linewidth=2, color='blue')

# Set plot properties
plt.title('Traveling Salesman Problem')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Show the plot
plt.show()
