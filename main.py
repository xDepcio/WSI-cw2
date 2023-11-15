import random
import numpy as np
from cec2017.functions import f2, f13


def tournament_repr(population, pop_fitness):
    new_population = []
    pop_w_fit = zip(population, pop_fitness)
    sorted_pop_w_fit = sorted(pop_w_fit, key=lambda entry: entry[1], reverse=False)
    sorted_pop_w_fit_w_ranks = [
        (rank + 1, indiv, fit) for rank, (indiv, fit) in enumerate(sorted_pop_w_fit)
    ]
    for _ in range(len(sorted_pop_w_fit_w_ranks)):
        (_, i_indiv, i_fit), (_, j_indiv, j_fit) = random.choices(
            sorted_pop_w_fit_w_ranks,
            k=2,
            weights=[
                1 - rank / len(sorted_pop_w_fit_w_ranks)
                for rank, _, _ in sorted_pop_w_fit_w_ranks
            ],
        )
        if i_fit <= j_fit:
            new_population.append(i_indiv)
        else:
            new_population.append(j_indiv)

    return new_population


def mutate(population, mutation_magnitude: float = 0.1):
    mutated_population = []
    for individual in population:
        mutated_individual = np.array(
            [
                gene + np.random.normal(loc=0.0, scale=mutation_magnitude, size=None)
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
    BUDGET: int = 10000,
    CLIP_L: int = 100,
):
    iter_limit = BUDGET / population_size
    curr_iter = 0

    curr_population = init_population(population_size, dimenstionality, -CLIP_L, CLIP_L)
    curr_pop_fitness = [target_function(individual) for individual in curr_population]
    best_indiv, best_fitness = min(
        zip(curr_population, curr_pop_fitness), key=lambda x: x[1]
    )

    while curr_iter < iter_limit:
        new_population = tournament_repr(curr_population, curr_pop_fitness)
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
        curr_pop_fitness = new_pop_fitness

    return best_indiv, best_fitness


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
        all_bests.append(best_fitness)

    print(
        f"F: {tested_func.__name__}, P_SIZE: {population_size}, MUT: {mutation_magnitude}"
    )
    print("AVG | MIN | MAX | STD")
    print(np.mean(all_bests), min(all_bests), max(all_bests), np.std(all_bests), "\n")


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
    ]
    for test in tests:
        main(*test)
