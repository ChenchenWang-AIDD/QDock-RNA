"""QUBO conversion, integer quantization and audited column split14.

Kaiwu private APIs are isolated here; importing this module does not load the SDK.
"""
import importlib
import time

import numpy as np


def qubo_to_ising(qubo_dict):
    """
    Convert a QUBO dictionary to Kaiwu-style Ising matrix via pure NumPy.

    Conversion formula:
    1. Q_sym = (Q + Q.T) / 8
    2. J[i,j] = Q_sym[i,j] (i≠j)
    3. J[aux,i] = J[i,aux] = sum_j(Q_sym[i,j]) + Q_sym[i,i]
    4. J = -J

    Args:
        qubo_dict: QUBO dictionary {(var1, var2): coeff}

    Returns:
        J: Ising coupling matrix (n+1 x n+1), last index is aux variable
        var_names: sorted variable name list
    """
    # Extract variable names and build index mapping.
    var_names = sorted(set(v for k in qubo_dict for v in k))
    n = len(var_names)
    var_idx = {name: i for i, name in enumerate(var_names)}

    # QUBO dict -> upper-triangular matrix.
    Q = np.zeros((n, n), dtype=np.float64)
    for (vi, vj), coef in qubo_dict.items():
        i, j = var_idx[vi], var_idx[vj]
        if i <= j:
            Q[i, j] += coef
        else:
            Q[j, i] += coef

    # QUBO -> Ising (consistent with kaiwu qubo_matrix_to_ising_matrix).
    Q_sym = (Q + Q.T) / 8
    diag = np.diag(Q_sym).copy()
    Q_off = Q_sym.copy()
    np.fill_diagonal(Q_off, 0)

    # Build (n+1)x(n+1) Ising matrix; last row/col are aux couplings.
    J = np.zeros((n+1, n+1), dtype=np.float64)
    J[:-1, :-1] = Q_off
    aux_coupling = Q_off.sum(axis=0) + diag
    J[-1, :-1] = aux_coupling
    J[:-1, -1] = aux_coupling

    return -J, var_names


def quantize_qubo(qubo, names, decimals):
    """Aggregate unordered QUBO pairs, quantize ONCE, then convert exactly.

    Returned C is integer: J = info['ising_unit_original_energy'] * C.
    The unit includes GCD reduction; energies use E=-s.T J s.
    Both orientations of a dictionary pair, if present, are added BEFORE round.
    """
    if decimals not in (2, 3) or len(names) != len(set(names)) or not names:
        raise ValueError('Expected unique variable names and round2 or round3')
    index = {name: i for i, name in enumerate(names)}
    n, scale = len(names), 10 ** decimals
    upper = np.zeros((n, n), dtype=np.float64)
    seen = set()
    for (left, right), value in qubo.items():
        i, j = index[left], index[right]
        seen.update((left, right))
        if i > j:
            i, j = j, i
        upper[i, j] += value
    if seen != set(names) or not np.all(np.isfinite(upper)):
        raise ValueError('QUBO variable mismatch or nonfinite coefficients')
    if np.max(np.abs(upper)) * scale * (n + 2) >= 2 ** 52:
        raise ValueError('Integer coefficients/row sums would exceed exact float64 range')
    integers = np.rint(upper * scale).astype(np.int64)
    error = np.abs(upper - integers / scale)
    info = {'qubo_round_decimals': decimals, 'qubo_unit': 1 / scale,
            'max_qubo_coefficient_error': float(error.max()),
            'max_unary_error': float(np.diag(error).max()),
            'original_unary_min': float(np.diag(upper).min()),
            'original_unary_max': float(np.diag(upper).max()),
            'quantized_unary_min': float(np.diag(integers).min() / scale),
            'quantized_unary_max': float(np.diag(integers).max() / scale)}
    np.fill_diagonal(error, 0)
    info['max_pair_error'] = float(error.max())
    del error, upper
    diagonal = np.diag(integers).copy()
    np.fill_diagonal(integers, 0)
    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)
    matrix[:-1, :-1] = -integers
    matrix[:-1, :-1] += -integers.T
    matrix[:-1, -1] = matrix[:-1, :-1].sum(axis=0) - 2 * diagonal
    matrix[-1, :-1] = matrix[:-1, -1]
    recovered = 4 * (matrix[:-1, :-1].sum(axis=1) - matrix[:-1, -1])
    if not np.array_equal(recovered, 8 * diagonal):
        raise RuntimeError('Integer QUBO to Ising unary cancellation audit failed')
    common = int(np.gcd.reduce(np.abs(matrix).reshape(-1))) or 1
    matrix //= common
    unit = common / (8 * scale)
    max_integer = int(np.abs(matrix).max())
    info.update(qubo_variables=n, original_ising_size=n+1,
                integer_gcd_removed=common, ising_unit_original_energy=unit,
                maximum_integer_coefficient=max_integer,
                integer_signed_bits=max_integer.bit_length()+1,
                maximum_physical_coefficient=max_integer*unit,
                integer_conversion_audit_passed=True)
    return matrix, info

def audit_projection(solver_matrix, logical_matrix, starts, ends, atol=1e-7):
    """Check offdiag(L.T J_solver L) == offdiag(J_logical), without dense L.

    Diagonal differences are state-independent constants. This proves energy
    difference preservation only on the copy-consistent subspace.
    """
    if solver_matrix.shape != (int(ends[-1]) + 1,) * 2:
        raise ValueError("Solver matrix dimension does not match copy groups")
    max_error, diag_difference = 0.0, 0.0
    for i, (a, b) in enumerate(zip(starts, ends)):
        physical_row = np.sum(solver_matrix[a:b + 1], axis=0)
        projected = np.add.reduceat(physical_row, starts)
        difference = projected - logical_matrix[i]
        diag_difference += float(difference[i])
        difference[i] = 0.0
        max_error = max(max_error, float(np.max(np.abs(difference))))
    if max_error > atol:
        raise ValueError("Split projection does not preserve logical couplings: %.12g" % max_error)
    return {"offdiagonal_max_error": max_error, "tolerance": atol,
            "split_minus_logical_constant": -diag_difference,
            "reference_spin_included": True, "passed": True}


class SplitSizeLimitExceeded(RuntimeError):
    """Predicted expanded size exceeds the limit, before expanded allocation."""

    def __init__(self, actual_size, maximum_size, plan):
        self.actual_size = int(actual_size)
        self.maximum_size = int(maximum_size)
        self.plan = plan
        super().__init__(f'planned final Ising size {actual_size} exceeds limit {maximum_size}')


def column_split14(integer_matrix, max_size=None, progress=None):
    """Plan and build only the validated SDK 1.0.33a column expansion.

    Input is an integer lattice; min_increment=1 and the SDK's default
    penalty are deliberate. Do not substitute the SDK's auto two-plan path.
    """
    kw = importlib.import_module("kaiwu")
    if getattr(kw, "__version__", None) != "1.0.33a":
        raise RuntimeError("Column split adapter requires Kaiwu 1.0.33a")
    sdk = importlib.import_module("kaiwu.preprocess._precision_adaption_split")
    matrix = np.asarray(integer_matrix, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix)
            or not np.all(np.isfinite(matrix)) or not np.array_equal(matrix, np.rint(matrix))):
        raise ValueError("Expected a nonempty square integer-valued matrix")
    emit = progress or (lambda stage, payload: None)
    started = time.monotonic()
    # Preserve SDK arithmetic, including averaging before the absolute value.
    normalized = (matrix + matrix.T) / 2
    np.abs(normalized, out=normalized)
    normalized /= 8191
    normalized /= 1.
    emit("split_plan_start", {"strategy": "column_based", "original_size": len(matrix)})
    counts = np.asarray(sdk.get_split_num_by_column_sum(normalized, len(matrix)))
    if (counts.shape != (len(matrix),) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 1)):
        raise RuntimeError("Invalid SDK copy-count plan")
    del normalized
    size = sum(int(count) for count in counts)
    planning_seconds = time.monotonic() - started
    plan = {
        "strategy": "column_based", "planned_size": size,
        "one_dense_float64_gib": size * size * 8 / 1024**3,
        "maximum_copies": int(counts.max()), "reference_spin_copies": int(counts[-1]),
        "expanded_original_variables": int(np.count_nonzero(counts > 1)),
        "planning_seconds": planning_seconds,
    }
    emit("split_plan_ready", {"plan": plan})
    if max_size is not None and size > max_size:
        raise SplitSizeLimitExceeded(size, max_size, plan)
    started = time.monotonic()
    physical, restore = sdk._perform_precision_adaption_split(
        matrix, param_bit=14, min_increment=1., penalty=None,
        round_to_increment=True, is_column_based=True,
    )
    if len(physical) != size or not np.array_equal(np.diff(np.r_[-1, restore]), counts):
        raise RuntimeError("Constructed split matrix differs from its plan")
    diagnostics = {
        "sdk_version": kw.__version__, "plans": [plan], "selected_strategy": "column_based",
        "selected_size": size, "planning_seconds": planning_seconds,
        "build_seconds": time.monotonic() - started, "expanded_matrix_build_calls": 1,
        "selection_rule": "column only; pair planner not called",
        "selected_plan_matches_restore": True,
    }
    emit("selected_split_built", {"diagnostics": diagnostics})
    return physical, restore, diagnostics
