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

    def steady_state_selection(self, fitness, num_parents, population):

        """
        Selects the parents using the steady-state selection algorithm. So that the  selected parents mate to produce their offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

        parents = np.empty((num_parents, population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = population[fitness_sorted[parent_num], :].copy()

        return parents, fitness_sorted[:num_parents]

    def rank_selection(self, fitness, num_parents, population):

        """
        Selects the parents using the rank selection technique. So that the  selected parents mate to produce their offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, population.shape[1]))
        n = len(parents)
        # rank_sum = n * (n + 1) / 2
        for parent_num in range(num_parents):
            parents[parent_num, :] = population[fitness_sorted[parent_num], :].copy()

        return parents, fitness_sorted[:num_parents]

    def random_selection(self, fitness, num_parents, population):

        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """


        parents = np.empty((num_parents, population.shape[1]))


        rand_indices = np.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = population[rand_indices[parent_num], :].copy()

        return parents, rand_indices

    def tournament_selection(self, fitness, num_parents):

        """
        Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = np.empty((num_parents, population.shape[1]))

        parents_indices = []

        for parent_num in range(num_parents):
            rand_indices = np.random.randint(low=0.0, high=len(fitness), size=K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = np.where(K_fitnesses == np.max(K_fitnesses))[0][0]
            parents_indices.append(rand_indices[selected_parent_idx])
            parents[parent_num, :] = population[rand_indices[selected_parent_idx], :].copy()

        return parents, parents_indices

    def roulette_wheel_selection(self, fitness, num_parents, population):

        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = np.sum(fitness)
        if fitness_sum == 0:
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
        probs = fitness / fitness_sum
        probs_start = np.zeros(probs.shape,
                               dtype=np.float)  # An array holding the start values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape,
                             dtype=np.float)  # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

        parents = np.empty((num_parents, population.shape[1]))

        parents_indices = []
        # np.random.choice(4, 2, p=probs, replace=False)
        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = population[idx, :].copy()
                    parents_indices.append(idx)
                    break
        return parents, parents_indices

    def stochastic_universal_selection(self, fitness, num_parents, population):

        """
        Selects the parents using the stochastic universal selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = np.sum(fitness)
        if fitness_sum == 0:
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
        probs = fitness / fitness_sum
        probs_start = np.zeros(probs.shape,
                                  dtype=np.float64)  # An array holding the start values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape,
                                dtype=np.float64)  # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        pointers_distance = 1.0 / num_parents  # Distance between different pointers.
        first_pointer = np.random.uniform(low=0.0, high=pointers_distance, size=1)  # Location of the first pointer.

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, population.shape[1]))

        parents_indices = []

        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num * pointers_distance
            for idx in range(probs.shape[0]):
                if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                    parents[parent_num, :] = population[idx, :].copy()
                    parents_indices.append(idx)
                    break
        return parents, parents_indices

    def roulette_wheel_selection(self, population, fitness_values, k_tournemant=3):
        """Select parents for crossover using roulette wheel selection."""

        parents = []


        for i in range(2):
            parent_index = np.random.randint(low=0, high=self.population_size, size=k_tournemant)
            parent = np.where(fitness_values == np.max(fitness_values[parent_index]))

            parents.append(population[parent])
        return parents

    def tournament_selection(self, population, fitness_values):
        """Select parents for crossover using roulette wheel selection."""
        # cumulative_fitness = np.cumsum(fitness_values)
        probs = fitness_values / np.sum(fitness_values)

        parents = np.random.choice(len(probs), 2, p=probs, replace=False)

        return parents

    def rank_selection(self, population, fitness_values):
        # you
        parents = []
        for _ in range(2):
            index = np.argmax(fitness_values)
            individu  = fitness_values.pop(index)
            parents.append(population[index])
        fitness_values.insert(index, individu)
        return parents, fitness_values

    def crossover(self, parents):
        """Perform crossover to create a new offspring."""
        parent1, parent2 = parents[0], parents[1]
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
        # you should select the all parents at once !!!
        for generation in range(num_generations):
            fitness_values = self.evaluate_fitness(population)
            for i in range(self.population_size):
                parents = self.select_parents(population, fitness_values)
                # parents,fitness_values = self.rank_selection(population, fitness_values)
                offspring = self.crossover(parents)
                offspring = self.mutate(offspring)
                population.append(offspring)

            population = sorted(population, key=lambda x: self.fitness_function(x))[:self.population_size]

        best_feature_set = population[0]
        best_fitness = self.fitness_function(best_feature_set)
        return best_feature_set, best_fitness


