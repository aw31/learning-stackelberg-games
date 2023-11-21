from typing import Optional

import numpy as np
from scipy import optimize

EPSILON = 1e-9


class Polytope:
    """
    Polytope represented via constraints {x : Ax <= b}
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        assert b.shape == (A.shape[0], 1)

        self.dimension = A.shape[1]
        self.A = A
        self.b = b

    def contains(self, x: np.ndarray):
        assert x.shape == (self.dimension, 1)
        return (self.A @ x <= self.b).all()

    def intersect(self, other: "Polytope"):
        return Polytope(
            np.concatenate([self.A, other.A]), np.concatenate([self.b, other.b])
        )

    def sample_interior_point(self, epsilon=1e-5) -> Optional[np.ndarray]:
        """
        Solves for the Chebyshev center of the polytope
        """
        A_norms = np.linalg.norm(self.A, axis=1).reshape(-1, 1)
        A_chebyshev = np.concatenate([self.A, A_norms], axis=1)
        c_chebyshev = np.concatenate([np.zeros(self.dimension), -np.ones(1)])
        result = optimize.linprog(
            c=c_chebyshev,
            A_ub=A_chebyshev,
            b_ub=self.b,
            bounds=(None, None),
        )
        if result.success and result.x[-1] > epsilon:
            x = result.x[:-1].reshape(-1, 1)
            assert (self.A @ x <= self.b - EPSILON).all(), (result.x[-1], result.slack)
            return x
        else:
            return None


class SSG:
    """
    Stackelberg security game implementation
    """

    def __init__(
        self,
        n_targets: int,
        constraints: Polytope,
        leader_payoffs: np.ndarray,
        follower_payoffs: np.ndarray,
    ):
        assert constraints.dimension == n_targets
        assert leader_payoffs.shape == (2, n_targets)
        assert follower_payoffs.shape == (2, n_targets)
        assert (leader_payoffs[1] > leader_payoffs[0]).all()
        assert (follower_payoffs[1] < follower_payoffs[0]).all()

        self.n_targets = n_targets
        self.constraints = constraints
        self.leader_payoffs = leader_payoffs
        self.follower_payoffs = follower_payoffs
        self.call_count = 0

    def is_feasible(self, x: np.ndarray):
        """
        Checks feasibility of leader action x against constraints
        """
        assert x.shape == (self.n_targets, 1)
        return self.constraints.contains(x)

    def get_best_response(self, x: np.ndarray, boost_index=None, boost_epsilon=1e-5):
        """
        Implements best response oracle, returning follower's best response to leader action x
        """
        self.call_count += 1
        follower_payoffs = (
            self.follower_payoffs[1] * x.T + self.follower_payoffs[0] * (1.0 - x.T)
        ).squeeze()
        if boost_index is not None:
            follower_payoffs[boost_index] += boost_epsilon
        return np.argmax(follower_payoffs)

    def get_leader_payoff(self, x: np.ndarray):
        y = self.get_best_response(x)
        return self.leader_payoffs[1][y] * x[y] + self.leader_payoffs[0][y] * (
            1.0 - x[y]
        )
