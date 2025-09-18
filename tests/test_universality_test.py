import pytest
import jax
import jax.numpy as jnp
import numpy as np

from unicirc.universality_test import (
    _four_equals_test,
    jac_rank,
    assert_rank,
    rank_test,
    all_candidates_linear,
    filtered_candidates_linear,
    search_ansatze,
)

# ==============================================================================
## Tests for Helper Functions
# ==============================================================================

class TestHelperFunctions:
    """Tests for smaller helper and utility functions."""

    def test_four_equals_test_true(self):
        """Test _four_equals_test returns True for four equal consecutive connections."""
        connections = jnp.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        assert _four_equals_test(connections)

    def test_four_equals_test_false_different(self):
        """Test _four_equals_test returns False for different connections."""
        connections = jnp.array([[0, 1], [0, 2], [1, 2], [0, 1]])
        assert not _four_equals_test(connections)

    def test_jac_rank_full_rank(self):
        """Test jac_rank for a full-rank system."""
        # A simple linear function from C^3 to C^(3x3)
        def linear_fn(params):
            A = jnp.array([
                [[1, 2, 3], [4, 5, 6], [7, 8, 10]],
                [[2, 3, 1], [5, 6, 4], [8, 10, 7]],
                [[3, 1, 2], [6, 4, 5], [10, 7, 8]],
            ], dtype=jnp.complex128)
            return (A @ params).reshape((3, 3))

        params = jnp.array([0.1, 0.2, 0.3], dtype=jnp.complex128)
        jac_fn = jax.jacobian(linear_fn, holomorphic=True)
        # The flattened jacobian will have shape (9, 3) and full rank.
        rank, _ = jac_rank(jac_fn, params, tol=1e-8)
        assert rank == 3

    def test_jac_rank_deficient_rank(self):
        """Test jac_rank for a rank-deficient system. (Corrected)"""
        # Define a tensor where the third 'page' is a linear combination of the first two.
        # This ensures the resulting flattened 9x3 Jacobian will have rank 2.
        page_0 = jax.random.normal(jax.random.PRNGKey(1), (3, 3), dtype=jnp.complex128)
        page_1 = jax.random.normal(jax.random.PRNGKey(2), (3, 3), dtype=jnp.complex128)
        page_2 = page_0 + 2j * page_1  # Create a linear dependency

        T_deficient = jnp.stack([page_0, page_1, page_2], axis=-1)  # Shape (3, 3, 3)

        def linear_fn_deficient(params):
            # This correctly maps a 3-element param vector to a 3x3 matrix
            return T_deficient @ params

        params = jnp.array([0.1, 0.2, 0.3], dtype=jnp.complex128)
        jac_fn = jax.jacobian(linear_fn_deficient, holomorphic=True)

        # The Jacobian is T_deficient. When flattened to (9, 3), its column rank is 2.
        rank, _ = jac_rank(jac_fn, params, tol=1e-8)
        assert rank == 2

    def test_assert_rank_success(self):
        """Test that assert_rank passes when the rank is correct."""
        def fn(params):
            return jnp.eye(2) * params[0]

        jac_fn = jax.jacobian(fn, holomorphic=True)
        params = jnp.array([1.0], dtype=jnp.complex128)
        assert_rank(jac_fn, params, expected_rank=1, tol=1e-8)

    def test_assert_rank_failure(self):
        """Test that assert_rank fails when the rank is incorrect."""
        def fn(params):
            return jnp.eye(2) * params[0]

        jac_fn = jax.jacobian(fn, holomorphic=True)
        params = jnp.array([1.0], dtype=jnp.complex128)
        with pytest.raises(AssertionError, match="rank=1, expected_rank=2"):
            assert_rank(jac_fn, params, expected_rank=2, tol=1e-8)

# ==============================================================================
## Tests for Candidate Generators
# ==============================================================================

class TestCandidateGenerators:
    """Tests for the candidate generator functions."""

    def test_all_candidates_linear(self):
        """Test the all_candidates_linear generator."""
        num_wires, num_cz = 3, 2
        candidates, num_candidates = all_candidates_linear(num_wires, num_cz)

        expected_num = (num_wires - 1) ** num_cz
        assert num_candidates == expected_num

        expected_candidates = [
            ((0, 1), (0, 1)),
            ((0, 1), (1, 2)),
            ((1, 2), (0, 1)),
            ((1, 2), (1, 2)),
        ]
        assert list(candidates) == expected_candidates

    def test_filtered_candidates_linear(self):
        """Test the filtered_candidates_linear generator."""
        # Candidate to be filtered: four consecutive (0,1) connections
        bad_candidate_tuple = tuple([(0, 1)] * 4 + [(1, 2)])
        # Candidate that should pass
        good_candidate_tuple = ((0, 1), (1, 2), (0, 1), (1, 2), (0, 1))

        num_wires, num_cz = 3, 5
        candidates, num_candidates = filtered_candidates_linear(num_wires, num_cz)

        assert bad_candidate_tuple not in candidates
        assert good_candidate_tuple in candidates
        assert num_candidates == len(candidates)

# ==============================================================================
## Mocks and Fixtures for Main Test Functions
# ==============================================================================

@pytest.fixture
def mock_jac_rank_search(monkeypatch):
    """Fixture to mock the jac_rank function for search_ansatze."""
    mock_results = {}

    def mock_fn(mat_jac_fn, params, tol=1e-6):
        # Create a hashable representation of the partial function's connections
        connections_tuple = tuple(map(tuple, mat_jac_fn.keywords['connections'].tolist()))
        # Default to a non-universal rank if not specified
        return mock_results.get(connections_tuple, 0), None

    monkeypatch.setattr("unicirc.universality_test.jac_rank", mock_fn)
    return mock_results

# ==============================================================================
## Tests for search_ansatze
# ==============================================================================

class TestSearchAnsatze:
    """Tests for the main search_ansatze function."""

    def test_search_ansatze_invalid_wires(self):
        """Test that search_ansatze raises an error for an unsupported number of wires."""
        with pytest.raises(AssertionError, match="Currently only three or four qubits are supported"):
            search_ansatze(num_wires=2, num_cz=1)

    def test_search_ansatze_finds_first_solution(self, mock_jac_rank_search):
        """Test it finds a single universal ansatz and stops."""
        num_wires, num_cz = 3, 9
        dim = 4**num_wires - 1

        universal_connections = tuple([(0, 1)] * num_cz)
        non_universal_connections = tuple([(0, 2)] * num_cz)

        mock_jac_rank_search[universal_connections] = dim
        mock_jac_rank_search[non_universal_connections] = dim - 1

        candidates = [non_universal_connections, universal_connections] * 2
        def candidates_fn(nw, nc):
            return (c for c in candidates), len(candidates)

        result = search_ansatze(
            num_wires=num_wires, num_cz=num_cz, candidates_fn=candidates_fn, all_solutions=False
        )
        assert result is not None
        assert np.array_equal(result, np.array(universal_connections))

    def test_search_ansatze_returns_none(self, mock_jac_rank_search):
        """Test it returns None when no universal ansatz is found."""
        num_wires, num_cz = 3, 1
        dim = 4**num_wires - 1

        mock_jac_rank_search[((0, 1),)] = dim - 5
        mock_jac_rank_search[((1, 2),)] = dim - 10

        def candidates_fn(nw, nc):
            cands = [((0, 1),), ((1, 2),)]
            return (c for c in cands), len(cands)

        result = search_ansatze(num_wires=num_wires, num_cz=num_cz, candidates_fn=candidates_fn)
        assert result is None

    def test_search_ansatze_finds_all_solutions(self, mock_jac_rank_search):
        """Test it finds all universal ansatze when all_solutions is True."""
        num_wires, num_cz = 3, 9
        dim = 4**num_wires - 1

        universal_1 = tuple([(0, 1)] * num_cz)
        universal_2 = tuple([(1, 2)] * num_cz)
        non_universal = tuple([(0, 2)] * num_cz)

        mock_jac_rank_search[universal_1] = dim
        mock_jac_rank_search[universal_2] = dim
        mock_jac_rank_search[non_universal] = dim - 1

        candidates = [universal_1, non_universal, universal_2]
        def candidates_fn(nw, nc):
            return (c for c in candidates), len(candidates)

        solutions = search_ansatze(
            num_wires=num_wires, num_cz=num_cz, candidates_fn=candidates_fn, all_solutions=True
        )
        assert len(solutions) == 2
        assert np.array_equal(solutions[0], np.array(universal_1))
        assert np.array_equal(solutions[1], np.array(universal_2))

    def test_search_ansatze_reduction_test(self, mock_jac_rank_search, capsys):
        """Test that the reduction_test correctly filters candidates."""
        num_wires, num_cz = 3, 4

        # This candidate should be filtered by the reduction test
        reducible_candidate = ((0, 1), (0, 1), (0, 1), (0, 1))
        candidated = [reducible_candidate]

        def candidates_fn(nw, nc):
            return (c for c in candidated), len(candidated)

        # jac_rank should not even be called for the reducible candidate
        mock_jac_rank_search[reducible_candidate] = 4**num_wires - 1

        search_ansatze(
            num_wires, num_cz, reduction_test=True, candidates_fn=candidates_fn
        )

        captured = capsys.readouterr()
        # Check the print output to confirm it was discarded by the correct test
        assert "Discarded 1/1 candidates based on the reduction test" in captured.out
