from typing import Callable

import numpy as np
from scipy import stats

from security_games.utils import EPSILON


def find_boundary_point(
    oracle: Callable[[np.ndarray], bool], p: np.ndarray, x: np.ndarray, iters=100
):
    """
    Finds a point on the boundary of a polytope given by oracle, given a point p in
    the set and a point x outside the set
    """
    assert oracle(p) and not oracle(x)
    midpoint = (p + x) / 2
    for _ in range(iters):
        if oracle(midpoint):
            p = midpoint
        else:
            x = midpoint
        midpoint = (p + x) / 2
    return x


def sample_from_spherical_slice(v: np.ndarray, c: float):
    """
    Sample a point from the spherical slice of inner product c centered at v
    """
    v_norm = v / np.linalg.norm(v)
    x = np.random.randn(v.shape[0], 1)
    x -= np.sum(x * v_norm) * v_norm
    x_norm = x / np.linalg.norm(x)

    sample = -c * v_norm + np.sqrt(1 - c**2) * x_norm
    assert np.abs(np.linalg.norm(sample) - 1.0) < EPSILON

    return sample


def _get_simplex_vertices(dimension: int):
    alpha = 1.0 / dimension * (1 - 1.0 / np.sqrt(dimension + 1))
    beta = 1.0 / np.sqrt(dimension + 1)

    simplex_vertices = [-beta * np.ones((dimension, 1))]
    for i in range(dimension):
        e_i = np.zeros((dimension, 1))
        e_i[i] = 1
        simplex_vertices.append(e_i - alpha * np.ones((dimension, 1)))

    return simplex_vertices


def find_hyperplane(
    oracle: Callable[[np.ndarray], bool],
    p: np.ndarray,
    iters=100,
    epsilon=1e-8,
):
    """
    Given a point p on the boundary of a halfspace (locally given by oracle), find a hyperplane
    that separates p from the halfspace.

    This method first samples a randomly rotated simplex, and then finds the intersection of the
    edges of the simplex with the boundary of the halfspace. The hyperplane is then found by
    performing an SVD on the boundary points.
    """

    random_rotation = stats.special_ortho_group.rvs(p.shape[0])
    simplex_vertices = _get_simplex_vertices(p.shape[0])

    rotated_simplex_vertices = [random_rotation @ v for v in simplex_vertices]
    sentinel_points = [p + v * epsilon for v in rotated_simplex_vertices]
    sentinel_labels = [oracle(x) for x in sentinel_points]

    true_point = sentinel_points[sentinel_labels.index(True)]
    false_point = sentinel_points[sentinel_labels.index(False)]

    boundary_points = []
    for x, x_label in zip(sentinel_points, sentinel_labels):
        if x_label:
            boundary_point = find_boundary_point(oracle, x, false_point, iters=iters)
        else:
            boundary_point = find_boundary_point(oracle, true_point, x, iters=iters)
        boundary_points.append(boundary_point)

    boundary_points = np.hstack(boundary_points).T

    boundary_points -= p.T
    boundary_points /= np.linalg.norm(boundary_points, axis=1)[:, np.newaxis]

    _, S, V = np.linalg.svd(boundary_points)
    hyperplane = V[-1, :]

    assert S[-1] / S[0] < 1e-2, (S, p)

    hyperplane = V[-1, :][:, np.newaxis]
    if oracle(p + hyperplane * epsilon):
        hyperplane = -hyperplane
    return hyperplane
