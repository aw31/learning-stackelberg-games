import pprint

import numpy as np
import pandas as pd

from security_games.clinch import clinch
from security_games.utils import SSG, Polytope


def test_clinch(n_targets=3):
    dimension = n_targets
    simplex_A = np.concatenate([-np.eye(dimension), np.ones((1, dimension))])
    simplex_b = np.concatenate([np.zeros((dimension, 1)), np.ones((1, 1))])
    simplex = Polytope(simplex_A, simplex_b)

    values = [np.random.uniform(0.9995, 1) for _ in range(n_targets)]
    game = SSG(
        n_targets,
        simplex,
        np.array([[0.0, value] for value in values]).T,
        np.array([[value, 0.0] for value in values]).T,
    )

    _ = clinch(game, clinch_simplex=True, epsilon=1e-8)
    return game.call_count


if __name__ == "__main__":
    results = []
    for i in range(5, 101, 5):
        print(f"Testing n_targets={i}")
        call_count = test_clinch(i)
        results.append((i, call_count))
    pprint.pprint(results)

    df = pd.DataFrame(results, columns=["n_targets", "call_count"])
    df.to_json("results_clinch_1.jsonl", orient="records", lines=True)
