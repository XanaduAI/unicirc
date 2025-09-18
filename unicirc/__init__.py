import jax

jax.config.update("jax_enable_x64", True)
from .universal_ansatze import (
    ansatz_specs,
    make_ansatz,
    make_tape_from_ansatz,
)
from .matrix import matrix_v1, matrix_v2, matrix_v3, matrix_v2_partial
from .optimization import (
    compile,
    make_cost_fn,
    make_optimization_run,
    compile_adapt,
    sample_from_group,
)
from .count_clifford import count_clifford
from .universality_test import (
    rank_test,
    jac_rank,
    search_ansatze,
    all_candidates_linear,
    filtered_candidates_linear,
)
