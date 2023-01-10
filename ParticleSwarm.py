import numpy as np

class Particle:
    def _init_(self, num_dimensions, lower_bounds, upper_bounds, c1, c2, inertia_weight):
        self.num_dimensions = num_dimensions
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.c1 = c1
        self.c2 = c2
        self.inertia_weight = inertia_weight
        self.position = np.random.uniform(lower_bounds, upper_bounds, num_dimensions)
        self.velocity = np.zeros(num_dimensions)
        self.best_position = self.position
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position):
        r1 = np.random.uniform(0, 1, self.num_dimensions)
        r2 = np.random.uniform(0, 1, self.num_dimensions)
        cognitive_component = self.c1 * r1 * (self.best_position - self.position)
        social_component = self.c2 * r2 * (global_best_position - self.position)
        self.velocity = self.inertia_weight * self.velocity + cognitive_component + social_component

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, self.lower_bounds, self.upper_bounds)

    def evaluate(self, fitness_function):
        self.fitness = fitness_function(self.position)



if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position


class PSO:
    def _init_(self, num_particles, num_dimensions, lower_bounds, upper_bounds, c1, c2, inertia_weight, max_iterations):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.c1 = c1
        self.c2 = c2
        self.inertia_weight = inertia_weight
        self.max_iterations = max_iterations
        self.particles = [Particle(num_dimensions, lower_bounds, upper_bounds, c1, c2, inertia_weight) for _ in range(num_particles)]
        self.global_best_position = np.zeros(num_dimensions)
        self.global_best_fitness = float('inf')

    def optimize(self, fitness_function):
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                particle.evaluate(fitness_function)
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness