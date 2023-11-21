import numpy as np

from security_games.utils import SSG, Polytope


def conserve_mass(game: SSG, x: np.ndarray, x_lower: np.ndarray, epsilon: float):
    for i in range(game.n_targets):
        lb, ub = x_lower[i], x[i]
        while ub - lb > epsilon:
            m = (lb + ub) / 2
            x[i] = m
            if game.get_best_response(x) == i:
                lb = m
            else:
                ub = m
        x[i] = lb
    return x


def step_hit_and_run_mcmc(P: Polytope, x: np.ndarray, z: np.ndarray, epsilon: float):
    if not P.contains(x - epsilon * z) and not P.contains(x + epsilon * z):
        return x

    lb, ub = -epsilon, epsilon
    while P.contains(x + lb * z):
        lb *= 2
    while P.contains(x + ub * z):
        ub *= 2

    w = np.random.uniform(lb, ub)
    while not P.contains(x + w * z):
        w = np.random.uniform(lb, ub)

    return x + w * z


def sample_approximate_centroid(
    P: Polytope, x_lb: np.ndarray, active_set: set[int], epsilon, iters=3000
):
    A_augmented = np.concatenate((P.A, -np.eye(P.dimension)), axis=0)
    b_augmented = np.concatenate((P.b, -x_lb), axis=0)
    P_augmented = Polytope(A_augmented, b_augmented)
    inactive_mask = np.array([i not in active_set for i in range(P.dimension)])

    x = x_lb.copy()

    incr = 1e-7
    while P.contains(x + incr * np.ones(x.shape)):
        x += incr * np.ones(x.shape)
        incr *= 2

    iterates = []
    for _ in range(iters):
        z = np.random.normal(size=x.shape)
        z[inactive_mask] = 0.0
        z /= np.linalg.norm(z)

        x = step_hit_and_run_mcmc(P_augmented, x, z, epsilon)
        iterates.append(x.copy())

    return np.mean(iterates, axis=0)


def clinch(game: SSG, epsilon: float = 1e-3, clinch_simplex=False):
    game.call_count = 0

    if clinch_simplex:
        assert np.allclose(
            game.constraints.A,
            np.concatenate([-np.eye(game.n_targets), np.ones((1, game.n_targets))]),
        )
        assert np.allclose(
            game.constraints.b,
            np.concatenate([np.zeros((game.n_targets, 1)), np.ones((1, 1))]),
        )

    active_set = set(range(game.n_targets))
    x_lb = np.zeros((game.n_targets, 1))

    x = x_lb
    y = game.get_best_response(x)
    while y in active_set:
        if not clinch_simplex:
            x = sample_approximate_centroid(game.constraints, x_lb, active_set, epsilon)
        else:
            remainder = 1 - np.sum(x_lb)
            x = x_lb + remainder * np.ones(x_lb.shape) / game.n_targets
        y = game.get_best_response(x)
        x_lb[y] = x[y]

        for i in range(game.n_targets):
            e_i = np.zeros(x_lb.shape)
            e_i[i] = 1.0
            if not game.constraints.contains(x_lb + epsilon * e_i):
                active_set.remove(i)

    result = conserve_mass(game, x, x_lb, epsilon)

    print(f"Result: {result}")
    print(f"Queries: {game.call_count}")

    return result
