from typing import Optional

import numpy as np
from scipy import optimize

from security_games.security_search_utils import find_boundary_point, find_hyperplane
from security_games.utils import SSG, Polytope


def refine_polytope(
    game: SSG, i: int, P: Polytope, p: np.ndarray, x: np.ndarray, iters=100
):
    oracle = lambda x: game.get_best_response(x) == i
    y = find_boundary_point(oracle, p, x, iters=iters)
    v = find_hyperplane(oracle, y)
    return P.intersect(Polytope(v.T, v.T @ y))


def security_search(game: SSG, epsilon: float = 1e-8):
    game.call_count = 0
    upper_bounds = [game.constraints for _ in range(game.n_targets)]
    completed = [False for _ in range(game.n_targets)]
    points: list[Optional[np.ndarray]] = [None for _ in range(game.n_targets)]

    init_point = game.constraints.sample_interior_point()
    assert init_point is not None
    init_i = game.get_best_response(init_point)
    points[init_i] = init_point

    z = None
    while not all([completed[i] or (points[i] is None) for i in range(game.n_targets)]):
        done = False
        while not done:
            done = True
            for i in range(game.n_targets):
                P, p = upper_bounds[i], points[i]
                if p is None:
                    continue
                for j in range(i + 1, game.n_targets):
                    Q, q = upper_bounds[j], points[j]
                    if q is None:
                        continue
                    x = P.intersect(Q).sample_interior_point()
                    if x is not None:
                        done = False
                        k = game.get_best_response(x)

                        if k == i:
                            upper_bounds[j] = refine_polytope(game, j, Q, q, x)
                        if k == j:
                            upper_bounds[i] = refine_polytope(game, i, P, p, x)

                        if points[k] is None:
                            print(f"Found new best response {k}")
                            points[k] = x

        for i in range(game.n_targets):
            if points[i] is None or completed[i]:
                continue

            e_i = np.zeros((game.n_targets, 1))
            e_i[i] = 1
            A_norms = np.linalg.norm(upper_bounds[i].A, axis=1).reshape(-1, 1)
            result = optimize.linprog(
                c=-e_i,
                A_ub=upper_bounds[i].A,
                b_ub=upper_bounds[i].b - epsilon * A_norms,
                bounds=(None, None),
                method="simplex",
            )
            assert result.success, A_norms
            p = result.x.reshape(-1, 1)
            for j in range(game.n_targets):
                if points[j] is None:
                    p[j] = 0
            k = game.get_best_response(p, boost_index=i)
            if k == i:
                completed[i] = True
                z = p
            else:
                assert points[k] is None, (i, k)
                print(f"Found new best response {k}")
                points[k] = p
                break

    print(f"Result: {z}")
    print(f"Queries: {game.call_count}")

    return z
