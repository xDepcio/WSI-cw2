import functools
import random
import numpy as np
from torch import le

from cec2017.functions import f2, f13


def tournament_selection(population, pop_fitness, tournament_size: int = 2):
    selected_individuals = []
    sorted_population_w_fit = sorted(
        zip(population, pop_fitness), key=lambda entry: entry[1], reverse=False
    )
    pop_with_ranks = [
        (rank + 1, individual)
        for rank, (individual, _) in enumerate(sorted_population_w_fit)
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
        winner = random.choices(
            [indiv for _, indiv, _ in tournament_probabilities],
            weights=[prob for _, _, prob in tournament_probabilities],
        )[0]
        selected_individuals.append(winner)

    return selected_individuals


def mutate(population, mutation_magnitude: float = 0.1):
    mutated_population = []
    for individual in population:
        mutated_individual = np.array(
            [
                gene + np.random.uniform(-mutation_magnitude, mutation_magnitude)
                for gene in individual
            ]
        )
        mutated_population.append(mutated_individual)
    return mutated_population


def init_population(
    population_size: int, individual_size: int, bottom_limit: float, upper_limit: float
):
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(bottom_limit, upper_limit, size=individual_size)
        population.append(individual)
    return population


def evolve_best(
    target_function,
    population_size: int = 100,
    mutation_magnitude: float = 0.1,
    dimenstionality: int = 2,
):
    BUDGET = 50000
    CLIP_L = 100
    iter_limit = BUDGET / population_size
    curr_iter = 0

    curr_population = init_population(population_size, dimenstionality, -CLIP_L, CLIP_L)
    curr_pop_fitness = [target_function(individual) for individual in curr_population]
    best_indiv, best_fitness = min(
        zip(curr_population, curr_pop_fitness), key=lambda x: x[1]
    )

    while curr_iter < iter_limit:
        new_population = tournament_selection(
            curr_population, curr_pop_fitness, tournament_size=2
        )
        new_population = np.clip(
            mutate(new_population, mutation_magnitude=mutation_magnitude),
            -CLIP_L,
            CLIP_L,
        )
        new_pop_fitness = [target_function(individual) for individual in new_population]
        new_best_indiv, new_best_fitness = min(
            zip(new_population, new_pop_fitness), key=lambda x: x[1]
        )
        if new_best_fitness < best_fitness:
            best_fitness = new_best_fitness
            best_indiv = new_best_indiv

        curr_iter += 1
        curr_population = new_population

    return best_indiv, best_fitness


def booth_function(individual):
    x, y = individual
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def main(
    tested_func=f2,
    population_size=100,
    mutation_magnitude=0.1,
    dimenstionality=2,
):
    tested_func = tested_func
    all_bests = []
    for _ in range(30):
        best_indiv, best_fitness = evolve_best(
            target_function=tested_func,
            population_size=population_size,
            mutation_magnitude=mutation_magnitude,
            dimenstionality=dimenstionality,
        )
        all_bests.append((best_indiv, best_fitness))

    avg_best_fitness = functools.reduce(
        lambda acc, curr: acc + curr, [tup[1] for tup in all_bests]
    ) / len(all_bests)
    all_worst_fitness = max([tup[1] for tup in all_bests])
    all_best_fitness = min([tup[1] for tup in all_bests])
    std_deviation = np.std([tup[1] for tup in all_bests])

    avg_point = functools.reduce(
        lambda acc, curr: acc + curr, [tup[0] for tup in all_bests]
    ) / len(all_bests)
    avg_point_fitness = tested_func(avg_point)

    print(
        f"func: {tested_func.__name__}, Population size: {population_size}, mutation magnitude: {mutation_magnitude}"
    )

    print(
        f"Fitness stats: AVG: {avg_best_fitness}, BEST: {all_best_fitness} WORST: {all_worst_fitness}, STD: {std_deviation}, AVG_X: {avg_point}, AVG_X_F: {avg_point_fitness}\n"
    )


if __name__ == "__main__":
    tests = [
        # f2 tests
        (f2, 2, 0.1, 10),
        (f2, 4, 0.1, 10),
        (f2, 8, 0.1, 10),
        (f2, 16, 0.1, 10),
        (f2, 32, 0.1, 10),
        (f2, 64, 0.1, 10),
        (f2, 128, 0.1, 10),
        (f2, 2, 0.5, 10),
        (f2, 4, 0.5, 10),
        (f2, 8, 0.5, 10),
        (f2, 16, 0.5, 10),
        (f2, 32, 0.5, 10),
        (f2, 64, 0.5, 10),
        (f2, 128, 0.5, 10),
        (f2, 2, 1, 10),
        (f2, 4, 1, 10),
        (f2, 8, 1, 10),
        (f2, 16, 1, 10),
        (f2, 32, 1, 10),
        (f2, 64, 1, 10),
        (f2, 128, 1, 10),
        (f2, 2, 3, 10),
        (f2, 4, 3, 10),
        (f2, 8, 3, 10),
        (f2, 16, 3, 10),
        (f2, 32, 3, 10),
        (f2, 64, 3, 10),
        (f2, 128, 3, 10),
        (f2, 2, 5, 10),
        (f2, 4, 5, 10),
        (f2, 8, 5, 10),
        (f2, 16, 5, 10),
        (f2, 32, 5, 10),
        (f2, 64, 5, 10),
        (f2, 128, 5, 10),
        (f2, 2, 10, 10),
        (f2, 4, 10, 10),
        (f2, 8, 10, 10),
        (f2, 16, 10, 10),
        (f2, 32, 10, 10),
        (f2, 64, 10, 10),
        (f2, 128, 10, 10),
        # f13 tests
        (f13, 2, 0.1, 10),
        (f13, 4, 0.1, 10),
        (f13, 8, 0.1, 10),
        (f13, 16, 0.1, 10),
        (f13, 32, 0.1, 10),
        (f13, 64, 0.1, 10),
        (f13, 128, 0.1, 10),
        (f13, 2, 0.5, 10),
        (f13, 4, 0.5, 10),
        (f13, 8, 0.5, 10),
        (f13, 16, 0.5, 10),
        (f13, 32, 0.5, 10),
        (f13, 64, 0.5, 10),
        (f13, 128, 0.5, 10),
        (f13, 2, 1, 10),
        (f13, 4, 1, 10),
        (f13, 8, 1, 10),
        (f13, 16, 1, 10),
        (f13, 32, 1, 10),
        (f13, 64, 1, 10),
        (f13, 128, 1, 10),
        (f13, 2, 3, 10),
        (f13, 4, 3, 10),
        (f13, 8, 3, 10),
        (f13, 16, 3, 10),
        (f13, 32, 3, 10),
        (f13, 64, 3, 10),
        (f13, 128, 3, 10),
        (f13, 2, 5, 10),
        (f13, 4, 5, 10),
        (f13, 8, 5, 10),
        (f13, 16, 5, 10),
        (f13, 32, 5, 10),
        (f13, 64, 5, 10),
        (f13, 128, 5, 10),
        (f13, 2, 10, 10),
        (f13, 4, 10, 10),
        (f13, 8, 10, 10),
        (f13, 16, 10, 10),
        (f13, 32, 10, 10),
        (f13, 64, 10, 10),
        (f13, 128, 10, 10),
        # other
        # (booth_function, 150, 1, 2),
        # (f2, 16, 0.5, 2),
    ]
    for test in tests:
        main(*test)
