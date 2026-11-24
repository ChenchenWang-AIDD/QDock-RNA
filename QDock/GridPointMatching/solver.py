"""QUBO round -> FastSA (optional split) or direct neal -> logical descent.

This is the manuscript core, not a batch script. No SDK is imported until
splitting/solving is requested. Every returned sample is retained.
"""
from dataclasses import dataclass
from importlib.metadata import version
import time

import numpy as np

from .postprocess import (
    copy_groups, logical_descent, restore_logical_states,
    ising_energies, normalize_solver_states,
)
from .precision import (
    audit_projection, qubo_to_ising, quantize_qubo, column_split14,
    SplitSizeLimitExceeded,
)


@dataclass(frozen=True)
class PrecisionConfig:
    qubo_round_decimals: int | None = 2
    precision_method: str = "none"
    precision: int = 14
    max_ising_size: int | None = 11999
    logical_postprocess: bool = True
    logical_tolerance: float = 1e-9
    logical_max_steps: int = 10000

    def validate(self):
        if isinstance(self.qubo_round_decimals, bool) or self.qubo_round_decimals not in (None, 2, 3):
            raise ValueError("qubo_round_decimals must be None, 2 or 3")
        if self.precision_method not in ("none", "split"):
            raise ValueError("Supported precision methods: none, split")
        if self.precision_method == "split" and (self.precision != 14 or self.qubo_round_decimals is None):
            raise ValueError("Reproduction split requires round2/round3 QUBO and signed 14 bits")
        if self.max_ising_size is not None and (
            isinstance(self.max_ising_size, bool) or int(self.max_ising_size) != self.max_ising_size
            or self.max_ising_size < 1
        ):
            raise ValueError("max_ising_size must be a positive integer or None")
        if not np.isfinite(self.logical_tolerance) or self.logical_tolerance < 0:
            raise ValueError("logical_tolerance must be finite and nonnegative")
        if (isinstance(self.logical_max_steps, bool)
                or int(self.logical_max_steps) != self.logical_max_steps
                or self.logical_max_steps < 0):
            raise ValueError("logical_max_steps must be a nonnegative integer")


@dataclass(frozen=True)
class FastSAConfig:
    alpha: float = 0.8
    iterations_per_t: int = 5
    size_limit: int = 100
    n_pos: int = 30
    seed: int = 2026
    process_num: int = 1
    thread_num: int = 128

    def optimizer_arguments(self):
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        for key in ("iterations_per_t", "size_limit", "n_pos", "process_num", "thread_num"):
            value = getattr(self, key)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(key + " must be a positive integer")
        return dict(initial_temperature=None, cutoff_temperature=None, alpha=self.alpha,
                    iterations_per_t=self.iterations_per_t,
                    size_limit=max(self.size_limit, self.n_pos * 3), rand_seed=self.seed,
                    process_num=self.process_num, thread_num=self.thread_num)


@dataclass(frozen=True)
class NealConfig:
    """Only num_reads is passed by default; annealing settings stay SDK defaults."""
    num_reads: int = 100
    seed: int | None = None

    def sampler_arguments(self):
        if isinstance(self.num_reads, bool) or int(self.num_reads) != self.num_reads or self.num_reads < 1:
            raise ValueError("num_reads must be a positive integer")
        arguments = {"num_reads": int(self.num_reads)}
        if self.seed is not None:
            if (isinstance(self.seed, bool) or int(self.seed) != self.seed
                    or not 0 <= self.seed < 2**31):
                raise ValueError("neal seed must be an integer in [0, 2**31)")
            arguments["seed"] = int(self.seed)
        return arguments


@dataclass
class IsingProblem:
    original_full: np.ndarray
    quantized_integer: np.ndarray | None
    solver_matrix: np.ndarray
    variable_names: list[str]
    restore_ends: np.ndarray
    metadata: dict


@dataclass
class SpinBranch:
    states: np.ndarray
    energies: np.ndarray
    qubo_energies: np.ndarray
    ranking_target: str


@dataclass
class SolveResult:
    physical_states: np.ndarray
    copy_conflicts: np.ndarray
    branches: dict[str, SpinBranch]
    logical_diagnostics: dict | None = None
    post_error: str | None = None
    solver_metadata: dict | None = None


def prepare_ising(qubo, config=None, progress=None, *, solver_type="neal"):
    if solver_type not in ("fastsa", "neal"):
        raise ValueError("solver_type must be fastsa or neal")
    config = config or PrecisionConfig(precision_method="none" if solver_type == "neal" else "split")
    config.validate()
    if solver_type == "neal" and config.precision_method != "none":
        raise ValueError("Direct neal QUBO solving requires precision_method='none'; split is unsupported")
    names = sorted({v for pair in qubo for v in pair})
    if not names:
        raise ValueError("Cannot solve an empty QUBO")
    if config.max_ising_size is not None and len(names) + 1 > config.max_ising_size:
        raise SplitSizeLimitExceeded(len(names) + 1, config.max_ising_size,
                                     {"strategy": "original-size lower bound"})
    original, original_names = qubo_to_ising(qubo)
    if names != original_names or not np.all(np.isfinite(original)):
        raise ValueError("Invalid QUBO variable order or coefficients")
    integer = None
    if config.qubo_round_decimals is not None:
        integer, meta = quantize_qubo(qubo, names, config.qubo_round_decimals)
        solver_matrix = integer
        if solver_type == "neal":
            # Match sample_qubo's physical coefficients, not the storage-only integer scale.
            solver_matrix = integer.astype(np.float64) * meta["ising_unit_original_energy"]
    else:
        solver_matrix = original.copy()
        meta = dict(qubo_round_decimals=None, ising_unit_original_energy=1.,
                    quantization_domain="none")
    restore = np.arange(len(original), dtype=np.int64)
    if config.precision_method == "split":
        solver_matrix, restore, diagnostics = column_split14(
            integer, max_size=config.max_ising_size, progress=progress)
        if (not np.all(np.isfinite(solver_matrix)) or np.max(np.abs(solver_matrix)) > 8191
                or not np.array_equal(solver_matrix, np.rint(solver_matrix))):
            raise RuntimeError("Split matrix is not signed 14-bit integer valued")
        starts, restore = copy_groups(restore, len(solver_matrix), len(original))
        meta.update(split_selection=diagnostics,
                    projection_audit=audit_projection(solver_matrix, integer, starts, restore),
                    split_min_increment=1, split_penalty_integer=8191,
                    split_penalty_original_energy=8191 * meta["ising_unit_original_energy"],
                    final_integer_14bit_passed=True)
    meta.update(quantization_domain="QUBO" if integer is not None else "none",
                precision_method=config.precision_method, precision=config.precision,
                original_ising_size=len(original), final_ising_size=len(solver_matrix),
                reference_spin_logical_index=len(names), restore_last_var_idx=restore,
                pre_precision_round_decimals=None,
                descent_target="original unrounded full-precision Ising",
                solver_temperature_policy="FastSA auto initial/cutoff; integer matrix input"
                if integer is not None else "FastSA auto initial/cutoff; full matrix input")
    meta["solver_type"] = solver_type
    if solver_type == "neal":
        meta.update(solver_energy_scale=1,
                    solver_temperature_policy="neal SDK defaults; QUBO in original energy units")
    return IsingProblem(original, integer, solver_matrix, names, restore, meta)


def original_qubo_energies(original, states, offset=0.):
    zero = -np.ones((1, len(original)), dtype=np.int8)
    zero[:, -1] = 1
    return ising_energies(original, states) - ising_energies(original, zero)[0] + float(offset)


def process_physical_states(problem, returned, config=None, offset=0.):
    """Offline array entry: no preparation, grid generation, SDK or sampling."""
    config = config or PrecisionConfig(
        qubo_round_decimals=problem.metadata.get("qubo_round_decimals", 2),
        precision_method=problem.metadata.get("precision_method", "split"),
    )
    config.validate()
    physical = normalize_solver_states(returned, expected_size=len(problem.solver_matrix))
    if len(physical) == 0:
        raise RuntimeError("Solver returned no states")
    starts, ends = copy_groups(problem.restore_ends, len(problem.solver_matrix),
                              len(problem.original_full))
    logical, conflicts = restore_logical_states(physical, starts, ends)
    raw = SpinBranch(logical, ising_energies(problem.solver_matrix, physical),
                     original_qubo_energies(problem.original_full, logical, offset),
                     "split physical Ising integer energy" if config.precision_method == "split"
                     else "solver Ising energy")
    result = SolveResult(physical, conflicts, {"raw": raw})
    if config.logical_postprocess:
        try:
            post = logical_descent(problem.original_full, logical,
                                   config.logical_tolerance, config.logical_max_steps)
            result.branches["logical_fullprecision"] = SpinBranch(
                post["states"], post["energies"],
                original_qubo_energies(problem.original_full, post["states"], offset),
                "original full-precision Ising energy")
            result.logical_diagnostics = {k: v for k, v in post.items() if k != "states"}
        except Exception as error:
            # A postprocessing failure must not erase usable raw solutions.
            result.post_error = type(error).__name__ + ": " + str(error)
    return result


def solve_fastsa(problem, fastsa=None, config=None, offset=0., optimizer=None, on_sample=None):
    fastsa = fastsa or FastSAConfig()
    arguments = fastsa.optimizer_arguments()
    injected = optimizer is not None
    if optimizer is None:
        import kaiwu as kw
        optimizer = kw.classical.FastSimulatedAnnealingOptimizer(**arguments)
    returned = optimizer.solve(problem.solver_matrix)
    if on_sample is not None:
        on_sample(returned, {"solver_type": "fastsa", "optimizer_injected": injected})
    return process_physical_states(problem, returned, config, offset)


def _neal_qubo_terms(problem, qubo):
    """Recover rounded coefficients from the exact integer conversion, without rounding twice."""
    n = len(problem.variable_names)
    if problem.quantized_integer is not None:
        integer = problem.quantized_integer
        gcd = problem.metadata["integer_gcd_removed"]
        twice = (integer[:-1, :-1].sum(axis=1)-integer[:-1, -1])*gcd
        if np.any(twice % 2):
            raise ValueError("Integer Ising -> rounded QUBO unary audit failed")
        diagonal = twice // 2
        row, col = np.nonzero(np.triu(integer[:-1, :-1], 1))
        values = -integer[row, col]*gcd
        scale = 10**problem.metadata["qubo_round_decimals"]
    else:
        index = {name: i for i, name in enumerate(problem.variable_names)}
        terms = {}
        for (left, right), value in qubo.items():
            i, j = sorted((index[left], index[right]))
            terms[i, j] = terms.get((i, j), 0.) + float(value)
        diagonal = np.array([terms.pop((i, i), 0.) for i in range(n)])
        pairs = [(i, j, value) for (i, j), value in terms.items() if value != 0]
        row = np.array([i for i, _, _ in pairs], dtype=np.int64)
        col = np.array([j for _, j, _ in pairs], dtype=np.int64)
        values = np.array([v for _, _, v in pairs], dtype=np.float64)
        scale = 1
    return diagonal, row, col, values, scale


def _neal_sample_energies(terms, binary):
    """Avoid the SDK's incremental energy cancellation; integers sum exactly after QUBO round."""
    diagonal, row, col, values, scale = terms
    if np.issubdtype(diagonal.dtype, np.integer):
        bound = len(diagonal)*int(np.max(np.abs(diagonal), initial=0))
        bound += len(values)*int(np.max(np.abs(values), initial=0))
        if 8*bound >= np.iinfo(np.int64).max:
            raise OverflowError("Exact QUBO energy sum may exceed int64")
    energies = np.array([diagonal[x == 1].sum() + values[(x[row] == 1) & (x[col] == 1)].sum()
                         for x in binary], dtype=diagonal.dtype)
    # E_Ising(x=0) = -sum(a)/2 - sum(b)/4 for E=-s.T J s.
    ising = (4*energies-2*diagonal.sum()-values.sum()).astype(np.float64)/(4*scale)
    return energies.astype(np.float64)/scale, ising


def solve_neal(problem, qubo, neal=None, config=None, offset=0., sampler=None, on_sample=None):
    """Sample the rounded/unrounded QUBO directly, then reuse common raw/post recovery.

    No Kaiwu import or licensing; no split; no FastSA temperature/budget settings.
    on_sample belongs to the caller and persists raw before energy/postprocessing audits.
    """
    if problem.metadata.get("solver_type") != "neal" or problem.metadata.get("precision_method") != "none":
        raise ValueError("Use prepare_ising(..., solver_type='neal') with precision_method='none'")
    if config is not None:
        config.validate()
        if (config.precision_method != "none"
                or config.qubo_round_decimals != problem.metadata.get("qubo_round_decimals")):
            raise ValueError("neal precision config disagrees with the prepared model")
    arguments = (neal or NealConfig()).sampler_arguments()
    from dwave.samplers import SimulatedAnnealingSampler
    import dimod

    terms = _neal_qubo_terms(problem, qubo)
    diagonal, row, col, values, scale = terms
    n = len(diagonal)
    Q = {(i, i): float(value)/scale for i, value in enumerate(diagonal)}
    Q.update(((int(i), int(j)), float(value)/scale) for i, j, value in zip(row, col, values))
    # Retain zero diagonals so even isolated variables survive the round/SDK conversion.
    spin_bqm = dimod.BinaryQuadraticModel.from_qubo(Q).change_vartype(dimod.SPIN)
    field_error = max(abs(spin_bqm.get_linear(i)+2*problem.solver_matrix[i, -1]) for i in range(n))
    coupling_error = max((abs(v+2*problem.solver_matrix[i, j])
                          for (i, j), v in spin_bqm.quadratic.items()), default=0.)
    if field_error > 1e-8 or coupling_error > 1e-12:
        raise ValueError("neal QUBO and saved solver Ising coefficients disagree")
    if spin_bqm.num_interactions != np.count_nonzero(np.triu(problem.solver_matrix[:-1, :-1], 1)):
        raise ValueError("neal QUBO and saved solver Ising edge counts disagree")
    del spin_bqm
    sampler = sampler if sampler is not None else SimulatedAnnealingSampler()
    started = time.perf_counter()
    sampled = sampler.sample_qubo(Q, **arguments)
    elapsed = time.perf_counter()-started
    del Q
    if set(sampled.variables) != set(range(n)):
        raise ValueError("neal returned missing/unexpected variable labels")
    labels = {label: i for i, label in enumerate(sampled.variables)}
    binary = sampled.record.sample[:, [labels[i] for i in range(n)]]
    returned = np.column_stack((2*binary-1, np.ones(len(binary), dtype=np.int8)))
    metadata = dict(solver_type="neal", entrypoint="SimulatedAnnealingSampler.sample_qubo",
                    call_arguments=arguments, solver_call_wall_s=elapsed,
                    versions={p: version(p) for p in ("numpy", "dimod", "dwave-samplers")},
                    response_info=sampled.info, reported_qubo_energies=sampled.record.energy.copy(),
                    num_occurrences=sampled.record.num_occurrences.copy(),
                    field_audit_max_error=field_error, coupling_audit_max_error=coupling_error,
                    energy_source="recomputed QUBO coefficients; SDK energy diagnostic only")
    if on_sample is not None:
        on_sample(returned, metadata)
    if (sampled.vartype != dimod.BINARY or not np.all((binary == 0) | (binary == 1))
            or len(binary) != arguments["num_reads"] or not np.all(sampled.record.num_occurrences == 1)):
        raise ValueError("Expected one unaggregated binary state per neal read")
    qubo_energy, ising_energy = _neal_sample_energies(terms, binary)
    matrix_error = float(np.max(np.abs(ising_energy-ising_energies(problem.solver_matrix, returned))))
    tolerance = max(1e-9, 32*np.finfo(float).eps*(n+1)*float(np.abs(problem.solver_matrix).sum()))
    metadata.update(recomputed_qubo_energies=qubo_energy+float(offset),
                    reported_energy_max_error=float(np.max(np.abs(qubo_energy-sampled.record.energy))),
                    matrix_energy_audit_max_error=matrix_error, matrix_energy_audit_tolerance=tolerance)
    if matrix_error > tolerance:
        raise ValueError("Recomputed QUBO and saved Ising energies disagree beyond roundoff bound")
    result = process_physical_states(problem, returned, config, offset)
    result.branches["raw"].energies = ising_energy
    result.branches["raw"].ranking_target = "solver Ising energy recomputed from QUBO coefficients"
    result.solver_metadata = metadata
    return result
