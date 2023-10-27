import functools
import random
import numpy as np


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
        # winner = np.random.choice(
        #     [indiv for _, indiv, _ in tournament_probabilities],
        #     p=[prob for _, _, prob in tournament_probabilities],
        # )
        winner = random.choices(
            [indiv for _, indiv, _ in tournament_probabilities],
            weights=[prob for _, _, prob in tournament_probabilities],
        )[0]
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


def evolve_best(
    target_function, population_size: int = 100, mutation_magnitude: float = 0.1
):
    BUDGET = 10000
    iter_limit = BUDGET / population_size
    curr_iter = 0

    curr_population = init_population(population_size, 2, -100, 100)
    curr_pop_fitness = [target_function(individual) for individual in curr_population]
    best_indiv, best_fitness = min(
        zip(curr_population, curr_pop_fitness), key=lambda x: x[1]
    )

    while curr_iter < iter_limit:
        new_population = tournament_selection(
            curr_population, curr_pop_fitness, tournament_size=2
        )
        new_population = mutate(new_population, mutation_magnitude=mutation_magnitude)
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


def main():
    tested_func = booth_function
    all_bests = []
    for _ in range(25):
        best_indiv, best_fitness = evolve_best(
            target_function=tested_func, population_size=100, mutation_magnitude=1
        )
        all_bests.append(best_indiv)

    print(all_bests)
    # avg_best = np.sum(all_bests) / len(all_bests)
    avg_best = functools.reduce(lambda acc, curr: acc + curr, all_bests)

    print("avg", avg_best)
    print(f"Average best X: {avg_best}, best fitness: {tested_func(avg_best)}")


if __name__ == "__main__":
    main()
