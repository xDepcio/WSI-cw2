import random
import numpy as np


def tournament_selection(population, function, tournament_size: int = 2):
    selected_individuals = []
    sorted_population = sorted(
        population, key=lambda indiv: function(indiv), reverse=True
    )
    pop_with_ranks = [
        (rank + 1, individual) for rank, individual in enumerate(sorted_population)
    ]

    for _ in range(len(population)):
        tournament = random.sample(pop_with_ranks, tournament_size)
        tournament_probabilities = [
            (
                rank,
                individual,
                1
                / len(population)
                * (
                    (len(population) - rank + 1) ** tournament_size
                    - (len(population) - rank) ** tournament_size
                ),
            )
            for rank, individual in tournament
        ]
        winner = np.random.choice(
            [indiv for _, indiv, _ in tournament_probabilities],
            p=[prob for _, _, prob in tournament_probabilities],
        )
        selected_individuals.append(winner)

    return selected_individuals


def mutate(population, mutation_magnitude: float = 0.1):
    mutated_population = []
    for individual in population:
        mutated_individual = [
            gene + random.uniform(-mutation_magnitude, mutation_magnitude)
            for gene in individual
        ]
        mutated_population.append(mutated_individual)
    return mutated_population


def init_population(
    population_size: int, individual_size: int, bottom_limit: float, upper_limit: float
):
    population = []
    for _ in range(population_size):
        individual = [
            random.uniform(bottom_limit, upper_limit) for _ in range(individual_size)
        ]
        population.append(individual)
    return population


def main(target_function, population_size: int = 100, mutation_magnitude: float = 0.1):
    BUDGET = 10000
    iter_limit = BUDGET / population_size
    curr_iter = 0

    population = init_population(population_size, 10, -100, 100)

    while curr_iter < iter_limit:
        population = tournament_selection(
            population, target_function, tournament_size=2
        )
        population = mutate(population, mutation_magnitude=mutation_magnitude)
        curr_iter += 1


def test():
    xd = np.array([4, 2, 3])
    xd2 = sorted(xd, key=lambda x: x, reverse=True)
    print(xd2)


test()
