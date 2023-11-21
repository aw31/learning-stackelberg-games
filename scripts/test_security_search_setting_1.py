import pprint

import numpy as np
import pandas as pd

from security_games.security_search import security_search
from security_games.utils import SSG, Polytope


def test_security_search(n_targets=3):
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

    _ = security_search(game)

    return game.call_count


if __name__ == "__main__":
    import multiprocessing

    pool = multiprocessing.Pool(8)
    call_count_results = []
    for i in range(5, 101, 5):
        print(f"Testing n_targets={i}")
        for _ in range(3):
            call_count_result = pool.apply_async(test_security_search, args=(i,))
            call_count_results.append((i, call_count_result))

    call_counts = []
    for i, call_count_result in call_count_results:
        try:
            call_count = call_count_result.get()
        except Exception as e:
            print(f"Error for n_targets={i}!\n{e}")
        else:
            print(f"n_targets={i} call_count={call_count}")
            call_counts.append((i, call_count))
            pprint.pprint(call_counts)

    pprint.pprint(call_counts)

    df = pd.DataFrame(call_counts, columns=["n_targets", "call_count"])
    df.to_json("results_security_search_1.jsonl", orient="records", lines=True)
