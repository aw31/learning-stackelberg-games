import numpy as np

from security_games.security_search_utils import _get_simplex_vertices, find_hyperplane
from security_games.utils import Polytope


def test_find_hyperplane(dimension, base_iters=60):
    simplex_A = np.concatenate([-np.eye(dimension), np.ones((1, dimension))])
    simplex_b = np.concatenate([np.zeros((dimension, 1)), np.ones((1, 1))])
    simplex = Polytope(simplex_A, simplex_b)
    p_simplex = np.ones((dimension, 1)) / dimension

    v_simplex = find_hyperplane(
        lambda x: bool(simplex.contains(x)), p_simplex, iters=dimension * base_iters
    )
    v_simplex_target = np.ones((dimension, 1)) / np.sqrt(dimension)
    simple_err = np.linalg.norm(v_simplex - v_simplex_target)
    assert simple_err < 1e-2, (dimension, simple_err, v_simplex)
    print(f"simplex_err: {simple_err}")

    cube_A = np.concatenate([-np.eye(dimension), np.eye(dimension)])
    cube_b = np.concatenate([np.zeros((dimension, 1)), np.ones((dimension, 1))])
    cube = Polytope(cube_A, cube_b)
    p_cube = np.concatenate([0.5 * np.ones((dimension - 1, 1)), np.zeros((1, 1))])

    v_cube = find_hyperplane(
        lambda x: bool(cube.contains(x)), p_cube, iters=dimension * base_iters
    )
    v_cube_target = -np.concatenate([np.zeros((dimension - 1, 1)), np.ones((1, 1))])
    cube_err = np.linalg.norm(v_cube - v_cube_target)
    assert cube_err < 1e-2, (dimension, cube_err, v_cube)
    print(f"cube_err: {cube_err}")


def test_sample_interior_point(dimension=3):
    simplex_A = np.concatenate([-np.eye(dimension), np.ones((1, dimension))])
    simplex_b = np.concatenate([np.zeros((dimension, 1)), np.ones((1, 1))])
    simplex = Polytope(simplex_A, simplex_b)
    x = simplex.sample_interior_point()
    assert x is not None and simplex.contains(x), x
    print(f"x: {x}")

    simplex_A = np.concatenate([np.eye(dimension), -np.ones((1, dimension))])
    simplex_b = np.concatenate([np.zeros((dimension, 1)), np.ones((1, 1))])
    simplex = Polytope(simplex_A, simplex_b)
    x = simplex.sample_interior_point()
    assert x is not None and simplex.contains(x), x
    print(f"x: {x}")

    cube_A = np.concatenate([-np.eye(dimension), np.eye(dimension)])
    cube_b = np.concatenate([np.zeros((dimension, 1)), np.ones((dimension, 1))])
    cube = Polytope(cube_A, cube_b)
    x = cube.sample_interior_point()
    print(f"x: {x}")


def test_get_simplex_vertices(dimension=4):
    vertices = _get_simplex_vertices(dimension)
    assert len(vertices) == dimension + 1

    dists = [np.linalg.norm(u - v) for u in vertices for v in vertices if u is not v]
    assert np.max(dists) - np.min(dists) < 1e-6

    print(f"simplex vertices: {vertices}")


if __name__ == "__main__":
    test_find_hyperplane(3)
    test_find_hyperplane(10)
    test_find_hyperplane(30)
    test_find_hyperplane(100)
    # test_find_hyperplane(300)

    test_sample_interior_point(10)
    test_get_simplex_vertices(4)
