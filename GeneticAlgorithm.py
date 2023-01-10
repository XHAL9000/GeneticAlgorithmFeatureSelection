import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, population_size, num_features, fitness_function, mutation_rate=0.01):
        self.population_size = population_size
        self.num_features = num_features
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        
    def initialize_population(self):
        """Initialize population randomly with binary encoded feature sets."""
        population = []
        for i in range(self.population_size):
            feature_set = np.random.randint(2, size=self.num_features)
            population.append(feature_set)
        return population
    
    def evaluate_fitness(self, population):
        """Evaluate the fitness of each feature set in the population."""
        fitness_values = []
        for feature_set in population:
            fitness = self.fitness_function(feature_set)
            fitness_values.append(fitness)
        return fitness_values
    
    def select_parents(self, population, fitness_values):
        """Select parents for crossover using roulette wheel selection."""
        cumulative_fitness = np.cumsum(fitness_values)
        parents = []
        for i in range(2):
            rand = random.uniform(0, cumulative_fitness[-1])
            parent_index = np.searchsorted(cumulative_fitness, rand)
            parents.append(population[parent_index])
        return parents
    
    def crossover(self, parents):
        """Perform crossover to create a new offspring."""
        parent1, parent2 = parents
        crossover_point = np.random.randint(1, self.num_features)
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return offspring
    
    def mutate(self, offspring):
        """Perform mutation on the offspring with a small probability."""
        for i in range(self.num_features):
            if random.uniform(0, 1) < self.mutation_rate:
                offspring[i] = 1 - offspring[i]
        return offspring
    
    def run(self, num_generations):
        """Run the genetic algorithm for the specified number of generations."""
        population = self.initialize_population()
        for generation in range(num_generations):
            fitness_values = self.evaluate_fitness(population)
            parents = self.select_parents(population, fitness_values)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            population.append(offspring)
            population = sorted(population, key=lambda x: self.fitness_function(x))[:self.population_size]
        
        best_feature_set = population[0]
        best_fitness = self.fitness_function(best_feature_set)
        return best_feature_set, best_fitness
