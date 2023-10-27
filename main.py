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
        # random.choices(
        #     population=[indiv for _, indiv, _ in tournament_probabilities],
        #     weights=[prob for _, _, prob in tournament_probabilities],
        # )[0]
        selected_individuals.append(winner)

    return selected_individuals


def test():
    xd = np.array([4, 2, 3])
    xd2 = sorted(xd, key=lambda x: x, reverse=True)
    print(xd2)


test()
