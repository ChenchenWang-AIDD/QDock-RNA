"""Physical-to-logical restoration, full-precision descent and rigid poses.

No SDK, molecular preparation or grid generation is called here.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseRecord:
    source_index: int
    energy_rank: int
    ranking_energy: float
    qubo_energy: float
    selected_variables: list[str]
    selection_counts: np.ndarray
    recovery_status: str
    coordinates: np.ndarray | None = None


@dataclass
class PoseBranch:
    name: str
    records: list[PoseRecord]
    # State rows stay in source order, NOT energy order.
    states: np.ndarray

    @property
    def poses(self):
        return [record.coordinates for record in self.records if record.coordinates is not None]


def normalize_solver_states(raw_states, expected_size=None):
    """Return a two-dimensional int8 matrix containing only -1/+1 spins."""
    states = np.asarray(raw_states)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2:
        raise ValueError("raw_states must be a one- or two-dimensional array")
    if expected_size is not None and states.shape[1] != int(expected_size):
        raise ValueError(
            "raw state width %d does not match Ising size %d"
            % (states.shape[1], int(expected_size))
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("raw_states contains non-finite values")

    rounded = np.rint(states)
    if not np.allclose(states, rounded, rtol=0.0, atol=1e-6):
        raise ValueError("raw_states must contain binary values or Ising spins")
    values = set(np.unique(rounded).astype(int).tolist())
    if values.issubset({-1, 1}):
        return rounded.astype(np.int8)
    if values.issubset({0, 1}):
        return (2 * rounded.astype(np.int8) - 1).astype(np.int8)
    raise ValueError("raw_states must contain only {-1, +1} or {0, 1}")


def ising_energies(matrix, states):
    """Evaluate ``-s.T @ J @ s`` for each state row."""
    matrix = np.asarray(matrix, dtype=np.float64)
    states = normalize_solver_states(states, expected_size=len(matrix))
    return -np.sum(states * (states @ matrix.T), axis=1)


def canonicalize_gauge(states, auxiliary_index=None):
    """Use the QUBO auxiliary spin as +1 without changing the Ising energy."""
    states = normalize_solver_states(states)
    if states.shape[1] == 0:
        raise ValueError("states must contain at least one spin")
    if auxiliary_index is None:
        auxiliary_index = states.shape[1] - 1
    auxiliary_index = int(auxiliary_index)
    if not 0 <= auxiliary_index < states.shape[1]:
        raise ValueError("auxiliary_index is outside the state matrix")
    return states * states[:, auxiliary_index : auxiliary_index + 1]

def copy_groups(ends, physical_size, logical_size):
    if ends is None:
        if physical_size != logical_size:
            raise ValueError("Missing restoration map for an expanded matrix")
        ends = np.arange(logical_size, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if (ends.shape != (logical_size,) or ends[0] < 0
            or np.any(np.diff(ends) <= 0) or ends[-1] != physical_size - 1):
        raise ValueError("Invalid contiguous-copy restoration map")
    return np.r_[0, ends[:-1] + 1], ends


def restore_logical_states(states, starts, ends):
    """Match installed SDK voting: strict majority, ties choose -1.

    Vote BEFORE gauge normalization: majority ties are not gauge invariant.
    Conflict information is retained rather than treating projection as descent.
    """
    states = normalize_solver_states(states, expected_size=int(ends[-1]) + 1)
    logical = np.empty((len(states), len(ends)), dtype=np.int8)
    conflicts = np.zeros_like(logical, dtype=bool)
    for i, (a, b) in enumerate(zip(starts, ends)):
        positive = np.count_nonzero(states[:, a:b + 1] > 0, axis=1)
        logical[:, i] = np.where(2 * positive > b - a + 1, 1, -1)
        conflicts[:, i] = (positive > 0) & (positive < b - a + 1)
    return canonicalize_gauge(logical, len(ends) - 1), conflicts


def logical_descent(matrix, states, tolerance=1e-9, max_steps=10000):
    """Steepest strict descent, fixed original reference spin (+1).

    No atom-level feasibility repair, swaps, resampling or RMSD information.
    One accepted move flips one ORIGINAL decision variable. Lowest index wins
    ties. Deduplication is computational only; every source row is returned.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("Expected a nonempty decision model with a reference spin")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("Logical matrix must be finite and symmetric")
    if tolerance < 0 or not np.isfinite(tolerance) or max_steps < 0:
        raise ValueError("Invalid descent limits")
    states = normalize_solver_states(states, expected_size=len(matrix))
    states = canonicalize_gauge(states, len(matrix) - 1)
    unique, inverse = np.unique(states, axis=0, return_inverse=True)
    energies_before = ising_energies(matrix, unique)
    fields = unique.astype(float) @ matrix.T
    diagonal = np.diag(matrix)
    out = unique.copy()
    predicted = energies_before.copy()
    steps = np.zeros(len(unique), dtype=np.int64)
    final_delta = np.empty(len(unique))
    for row, spin in enumerate(out):
        field = fields[row].copy()
        while True:
            deltas = 4.0 * spin[:-1] * field[:-1] - 4.0 * diagonal[:-1]
            index = int(np.argmin(deltas))
            decrease = float(deltas[index])
            if decrease >= -tolerance:
                final_delta[row] = decrease
                break
            if steps[row] >= max_steps:
                raise RuntimeError("Logical descent reached its step limit")
            old = int(spin[index])
            spin[index] = -old
            field -= 2.0 * old * matrix[:, index]
            predicted[row] += decrease
            steps[row] += 1
    energies_after = ising_energies(matrix, out)
    if not np.allclose(predicted, energies_after, rtol=1e-12, atol=1e-5):
        raise RuntimeError("Incremental energy drift")
    if np.any(energies_after > energies_before + 1e-7 + 1e-12 * np.abs(energies_before)):
        raise RuntimeError("Logical energy increased")
    if np.any(out[:, -1] != 1):
        raise RuntimeError("Reference spin changed")
    return {"states": out[inverse], "input_energies": energies_before[inverse],
            "energies": energies_after[inverse],
            "energy_drop": (energies_before - energies_after)[inverse],
            "steps": steps[inverse], "minimum_final_delta": final_delta[inverse],
            "n_unique_input": len(unique), "n_unique_output": len(np.unique(out, axis=0))}

def recover_branch(branch_name, branch, variable_names, ligand, box_coords):
    import prody

    records = []
    # Match the old (energy, source_index) ordering, keeping invalid ranks.
    order = sorted(range(len(branch.states)), key=lambda i: (branch.energies[i], i))
    for rank, index in enumerate(order, 1):
        selected = [name for name, value in zip(variable_names, branch.states[index, :-1])
                    if value > 0]
        counts = np.zeros(ligand.n, dtype=int)
        ligand_points, grid_points = [], []
        for name in selected:
            atom, grid = map(int, name.split("_"))
            counts[atom] += 1
            ligand_points.append(ligand.coords[atom])
            grid_points.append(box_coords[grid])
        coords = None
        status = "empty_assignment"
        if selected:
            try:
                transform = prody.superpose(np.asarray(ligand_points), np.asarray(grid_points))[1]
                coords = prody.applyTransformation(transform, ligand.coords)
                status = "ok"
            except Exception as error:
                status = "restore_failed:" + type(error).__name__
        records.append(PoseRecord(index, rank, float(branch.energies[index]),
                                  float(branch.qubo_energies[index]), selected, counts, status, coords))
    return PoseBranch(branch_name, records, branch.states)
